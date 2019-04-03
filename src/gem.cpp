#include "gem.h"

#include <fstream>
#include <cstdlib>

#include <glm/gtx/vector_angle.hpp>
#include <glm/gtx/matrix_decompose.hpp>
//#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace glm;
using namespace aruco;
using namespace Eigen;

namespace gem {
	PoseRunningAverager::PoseRunningAverager(int mP) : maxPoses(mP), numPoses(0) {
	}
	PoseRunningAverager::~PoseRunningAverager() {}

	void PoseRunningAverager::removeOldPose() {
		if (numPoses > 0) {
			poses.pop_front();
			numPoses--;
		}
	}

	void PoseRunningAverager::addNewPose(dvec3 pos, dquat ori) {
		while (numPoses >= maxPoses) {
			removeOldPose();
		}
		poses.push_back(make_tuple(true, pos, ori));
		numPoses++;
	}

	void PoseRunningAverager::addNullPose() {
		while (numPoses >= maxPoses) {
			removeOldPose();
		}
		poses.push_back(make_tuple(false, dvec3(), dquat()));
		numPoses++;
	}

	tuple<bool, dvec3, dquat> PoseRunningAverager::getAveragePose() {
		int numValidPoses = 0;
		dvec3 posInterp;
		dquat oriInterp;

		for (auto pose = poses.begin(); pose != poses.end(); pose++) {
			if (get<0>(*pose)) {
				dvec3 p = get<1>(*pose);
				dquat o = get<2>(*pose);

				if (numValidPoses == 0) {
					posInterp = p;
					oriInterp = o;
				}
				else {
					posInterp += p;
					oriInterp = slerp(o, oriInterp, 1.0 / (numValidPoses + 1));
				}
				numValidPoses++;
			}
		}
		if (numValidPoses > 0) {
			posInterp = posInterp / (double)numValidPoses;

			return make_tuple(true, posInterp, oriInterp);
		}
		else {
			return make_tuple(false, posInterp, oriInterp);
		}
	}

	GeometryExtendedMarker::GeometryExtendedMarker(float mSize, Dictionary dict) :
		submarkerSize(mSize),
		dictionary(dict),
		numSubmarkers((int)dict.size()),
		defaultTranslationRansacConfig(0.999, 0.4, 0.001, 0, 2000),
		defaultRotationRansacConfig(0.999, 0.4, 1, 0, 2000),
		baseSubmarker(0),
		calibrated(false),
		rangeNear(0),
		rangeFar(DBL_MAX),
		distMu(DBL_MAX),
		rotMu(DBL_MAX)
		//poseRunningAverageBufferSize(fBuffSize)
	{
		for (int i = 0; i < numSubmarkers; i++) {
			transformations.push_back(vector<InterMarkerTransformation>());
			for (int j = 0; j < numSubmarkers; j++) {
				transformations[i].push_back(InterMarkerTransformation());
			}

			//poseSmoother.push_back(PoseRunningAverager(poseRunningAverageBufferSize));
		}
	}

	GeometryExtendedMarker::~GeometryExtendedMarker() {
	}

	void GeometryExtendedMarker::setTrackingDistanceThreshold(double mu) {
		if (mu > 0) {
			distMu = mu;
		}
	}

	void GeometryExtendedMarker::setTrackingRotationThreshold(double mu) {
		if (mu > 0) {
			rotMu = mu;
		}
	}

	void GeometryExtendedMarker::setTrackingRange(double near, double far) {
		if (near > 0 && far > 0 && near < far) {
			rangeNear = near;
			rangeFar = far;
		}
	}

	const Dictionary GeometryExtendedMarker::getDictionary() {
		return dictionary;
	}

	pair<dvec3, dquat> getMarkerPose(aruco::Marker &marker) {
		double mPos[3], mOri[4];
		marker.OgreGetPoseParameters(mPos, mOri);
		dvec3 pos(mPos[0], mPos[1], mPos[2]);
		dquat ori = glm::normalize(dquat(mOri[0], mOri[1], mOri[2], mOri[3]));
		return pair<dvec3, dquat>(pos, ori);
	}

	dmat4 glMatFromVecQuat(dvec3 pos, dquat ori) {
		dmat4 trans = glm::translate(-pos) * glm::mat4_cast(ori) * glm::scale(dvec3(-1, -1, -1));
		return trans;
	}

	map<int, map<int, vector<tuple<dvec3, dquat, dvec3, dquat>>>> GeometryExtendedMarker::extractRelativePoses(
		vector<vector<Marker>> &calibrationSamples
		) {
		map<int, map<int, vector<tuple<glm::dvec3, glm::dquat, glm::dvec3, glm::dquat>>>> relativePoses;
		for (int i = 0; i < numSubmarkers; i++) {
			relativePoses.insert(pair<int, map<int, vector<tuple<dvec3, dquat, dvec3, dquat>>>>(
				i, map<int, vector<tuple<dvec3, dquat, dvec3, dquat>>>()
				));
			for (int j = i + 1; j < numSubmarkers; j++) {
				relativePoses.at(i).insert(pair<int, vector<tuple<dvec3, dquat, dvec3, dquat>>>(
					j, vector<tuple<dvec3, dquat, dvec3, dquat>>()
					));
			}
		}

		for (vector<vector<Marker>>::iterator it = calibrationSamples.begin(); it != calibrationSamples.end(); it++) {
			vector<Marker> markers = *it;
			int numMarkers = (int)markers.size();
			for (int i = 0; i < numMarkers; i++) {
				for (int j = i + 1; j < numMarkers; j++) {
					int fromId = glm::min(markers[i].id, markers[j].id);
					int toId = glm::max(markers[i].id, markers[j].id);

					if (fromId != toId) {
						pair<dvec3, dquat> fromPose = getMarkerPose(fromId == markers[i].id ? markers[i] : markers[j]);
						pair<dvec3, dquat> toPose = getMarkerPose(toId == markers[i].id ? markers[i] : markers[j]);

						relativePoses.at(fromId).at(toId).push_back(tuple<dvec3, dquat, dvec3, dquat>(
							fromPose.first, fromPose.second, toPose.first, toPose.second));
					}
				}
			}
		}

		return relativePoses;
	}

	void GeometryExtendedMarker::setBaseSubmarker(int id) {
		if (id >= 0 && id < numSubmarkers) {
			baseSubmarker = id;
		}
	}

	int GeometryExtendedMarker::getBaseSubmarker() {
		return baseSubmarker;
	}

	vector<Marker> GeometryExtendedMarker::filterOutliers(vector<aruco::Marker> &markers) {
		// first pass: remove those outside the tracking range
		vector<Marker> inRangeMarkers;
		vector<pair<dvec3, dquat>> inRangePoses;
		dvec3 origin(0, 0, 0);
		for (vector<Marker>::iterator it = markers.begin(); it != markers.end(); it++) {
			pair<dvec3, dquat> pose = getMarkerPose(*it);
			double dist = glm::distance(pose.first, origin);
			if (dist >= rangeNear && dist <= rangeFar) {
				inRangeMarkers.push_back(*it);
				inRangePoses.push_back(pose);
			}
		}

		if (!isCalibrated()) {
			return inRangeMarkers;
		}

		// second pass: remove those without neighbours
		vector<Marker> nonSoloMarkers;
		vector<bool> markerMask;
		for (size_t i = 0; i < inRangeMarkers.size(); i++) {
			markerMask.push_back(false);
		}
		for (size_t i = 0; i < inRangeMarkers.size(); i++) {
			int iId = inRangeMarkers[i].id;
			for (size_t j = i + 1; j < inRangeMarkers.size(); j++) {
				int jId = inRangeMarkers[j].id;
				if (iId != jId && transformations.at(jId).at(iId).valid) {
					dvec3 estPos = inRangePoses[j].first + inRangePoses[j].second * transformations.at(jId).at(iId).translation;
					dquat estOri = inRangePoses[j].second * transformations.at(jId).at(iId).rotation;
					double angle = glm::angle(inRangePoses[i].second * glm::inverse(estOri)) * 180.0 / glm::pi<double>();
					angle = glm::min(glm::abs(360.0 - angle), glm::abs(angle));
					//cout << angle << "?<" << rotMu << " "; // DBG
					if (glm::distance(estPos, inRangePoses[i].first) < distMu
						&& angle < rotMu) {
						//if (glm::distance(estPos, inRangePoses[i].first) < transformations.at(jId).at(iId).translationError
						//	&& angle < transformations.at(jId).at(iId).rotationError) {
						markerMask[i] = true;
						markerMask[j] = true;
					}
				}
			}
		}
		//cout << endl; // DBG
		for (size_t i = 0; i < inRangeMarkers.size(); i++) {
			if (markerMask[i]) {
				nonSoloMarkers.push_back(inRangeMarkers[i]);
			}
		}

		//cout << markers.size() << "->" << inRangeMarkers.size() << "->" << nonSoloMarkers.size() << endl;

		if (!nonSoloMarkers.empty()) {
			//cout << "good!" << endl; //DBG
			return nonSoloMarkers;
		}
		else {
			//cout << "bad!" << endl; // DBG
			return inRangeMarkers;
		}
	}

	bool GeometryExtendedMarker::computeRotationRANSAC(dquat &rotationOut, double &rotationErrorOut,
		vector<tuple<dvec3, dquat, dvec3, dquat>> &pairs, RansacConfig &config) {
		int totalNumberOfPoints = (int)pairs.size();
		if (totalNumberOfPoints < config.minInliers || totalNumberOfPoints < 1) {
			return false;
		}

		rotationErrorOut = DBL_MAX;

		int runs = config.numberOfRuns;
		while (runs-- > 0) {
			int iterations = config.getMaximumIterations(1);
			while (iterations-- > 0) {

				// random selection of point(s)
				tuple<dvec3, dquat, dvec3, dquat> chosenPoint = pairs[rand() % totalNumberOfPoints];

				dquat fromOri = get<1>(chosenPoint);
				dquat toOri = get<3>(chosenPoint);

				// solve model
				dquat chosenRot = glm::inverse(fromOri) * toOri;

				// find inliers and compute model
				dquat rotInterp;
				double rotError = 0;
				int inlierCount = 0;
				for (vector<tuple<dvec3, dquat, dvec3, dquat>>::iterator curPair = pairs.begin(); curPair != pairs.end(); curPair++) {
					fromOri = get<1>(*curPair);
					toOri = get<3>(*curPair);

					dquat estOri = fromOri * chosenRot;
					double angle = glm::angle(toOri * glm::inverse(estOri)) * 180.0 / glm::pi<double>();
					angle = glm::min(glm::abs(360.0 - angle), glm::abs(angle));
					if (angle < config.inlierTolerance) {
						// this is an inlier
						inlierCount++;

						if (inlierCount == 1) {
							rotInterp = glm::inverse(fromOri) * toOri;
						}
						else {
							rotInterp = glm::slerp(glm::inverse(fromOri) * toOri, rotInterp, 1.0 / inlierCount);
						}
						rotError += angle * angle;
					}
				}

				if (inlierCount >= config.minInliers) {
					double curError = glm::sqrt(rotError / (inlierCount - 1));
					if (curError < rotationErrorOut) {
						rotationOut = rotInterp;
						rotationErrorOut = curError;
					}
					break;
				}
			}
		}

		return rotationErrorOut != DBL_MAX;
	}

	bool GeometryExtendedMarker::computeTranslationRANSAC(dvec3 &translationOut, double &translationErrorOut,
		vector<tuple<dvec3, dquat, dvec3, dquat>> &pairs, RansacConfig &config, bool inverse = false // whether it is for from-to or to-from
		) {

		int totalNumberOfPoints = (int)pairs.size();
		if (totalNumberOfPoints < config.minInliers || totalNumberOfPoints < 1) {
			return false;
		}

		translationErrorOut = DBL_MAX;

		int runs = config.numberOfRuns;
		while (runs-- > 0) {
			int iterations = config.getMaximumIterations(1);
			while (iterations-- > 0) {

				// random selection of point(s)
				tuple<dvec3, dquat, dvec3, dquat> chosenPoint = pairs[rand() % totalNumberOfPoints];

				dvec3 fromPos, toPos;
				dquat fromOri;
				if (inverse) {
					fromPos = get<2>(chosenPoint);
					fromOri = get<3>(chosenPoint);
					toPos = get<0>(chosenPoint);
				}
				else {
					fromPos = get<0>(chosenPoint);
					fromOri = get<1>(chosenPoint);
					toPos = get<2>(chosenPoint);
				}

				// solve model
				dvec3 chosenTrans = glm::inverse(fromOri) * (toPos - fromPos);

				// find inliers and compute model
				dvec3 transInterp;
				double transError = 0;
				int inlierCount = 0;
				for (vector<tuple<dvec3, dquat, dvec3, dquat>>::iterator curPair = pairs.begin(); curPair != pairs.end(); curPair++) {
					if (inverse) {
						fromPos = std::get<2>(*curPair);
						fromOri = std::get<3>(*curPair);
						toPos = std::get<0>(*curPair);
					}
					else {
						fromPos = std::get<0>(*curPair);
						fromOri = std::get<1>(*curPair);
						toPos = std::get<2>(*curPair);
					}

					dvec3 estPos = fromPos + fromOri * chosenTrans;
					double dist = glm::distance(toPos, estPos);
					if (dist < config.inlierTolerance) {
						// this is an inlier
						inlierCount++;

						transInterp += glm::inverse(fromOri) * (toPos - fromPos);
						transError += dist * dist;
					}
				}

				if (inlierCount >= config.minInliers) {
					double curError = glm::sqrt(transError / (inlierCount - 1));
					if (curError < translationErrorOut) {
						translationOut = transInterp / (double)inlierCount;
						translationErrorOut = curError;
					}
					break;
				}
			}
		}

		return translationErrorOut != DBL_MAX;
	}

	dvec4 MagicStylus::threePointsToPlane(dvec3 &p1, dvec3 &p2, dvec3 &p3) {
		// plane.x * x + plane.y * y + plane.z * z + plane.w = 0;
		dvec3 p1p2 = p2 - p1;
		dvec3 p1p3 = p3 - p1;

		dvec3 normal = glm::normalize(glm::cross(p1p2, p1p3));
		double d = -glm::dot(normal, p1);



		return dvec4(normal, d);
	}

	dvec4 MagicStylus::fourPointsToSphere(dvec3 &p1, dvec3 &p2, dvec3 &p3, dvec3 &p4) {
		// (x - sphere.x)^2 + (y - sphere.y)^2 + (z - sphere.z)^2 = sphere.w^2
		// http://www.abecedarical.com/zenosamples/zs_sphere4pts.html
		double d1d1 = p1.x * p1.x + p1.y * p1.y + p1.z * p1.z;	// x_1^2 + y_1^2 + z_1^2
		double d2d2 = p2.x * p2.x + p2.y * p2.y + p2.z * p2.z;	// x_2^2 + y_2^2 + z_2^2
		double d3d3 = p3.x * p3.x + p3.y * p3.y + p3.z * p3.z;	// x_3^2 + y_3^2 + z_3^2
		double d4d4 = p4.x * p4.x + p4.y * p4.y + p4.z * p4.z;	// x_4^2 + y_4^2 + z_4^2

		double M11 = glm::determinant(dmat4x4(
			p1.x, p2.x, p3.x, p4.x,
			p1.y, p2.y, p3.y, p4.y,
			p1.z, p2.z, p3.z, p4.z,
			1, 1, 1, 1
			));
		double M12 = glm::determinant(dmat4x4(
			d1d1, d2d2, d3d3, d4d4,
			p1.y, p2.y, p3.y, p4.y,
			p1.z, p2.z, p3.z, p4.z,
			1, 1, 1, 1
			));
		double M13 = glm::determinant(dmat4x4(
			d1d1, d2d2, d3d3, d4d4,
			p1.x, p2.x, p3.x, p4.x,
			p1.z, p2.z, p3.z, p4.z,
			1, 1, 1, 1
			));
		double M14 = glm::determinant(dmat4x4(
			d1d1, d2d2, d3d3, d4d4,
			p1.x, p2.x, p3.x, p4.x,
			p1.y, p2.y, p3.y, p4.y,
			1, 1, 1, 1
			));
		double M15 = glm::determinant(dmat4x4(
			d1d1, d2d2, d3d3, d4d4,
			p1.x, p2.x, p3.x, p4.x,
			p1.y, p2.y, p3.y, p4.y,
			p1.z, p2.z, p3.z, p4.z
			));

		dvec4 sphere;
		double invM11 = 1 / M11;
		sphere.x = 0.5 * M12 * invM11;
		sphere.y = -0.5 * M13 * invM11;
		sphere.z = 0.5 * M14 * invM11;
		sphere.w = glm::sqrt(sphere.x * sphere.x + sphere.y * sphere.y + sphere.z * sphere.z - M15 * invM11);

		return sphere;
	}

	bool MagicStylus::computeStylusTipTranslationRANSAC(dvec3 &translationOut, double &translationErrorOut,
		dvec4 &sphere, vector<pair<dvec3, dquat>> &points, RansacConfig &config) {
		int totalNumberOfPoints = (int)points.size();
		if (totalNumberOfPoints < config.minInliers || totalNumberOfPoints < 1) {
			return false;
		}

		dvec3 sphereCenter(sphere.x, sphere.y, sphere.z);
		translationErrorOut = DBL_MAX;

		int runs = config.numberOfRuns;
		while (runs-- > 0) {
			int iterations = config.getMaximumIterations(1);
			while (iterations-- > 0) {
				// DBG
				//cout << "runs: " << runs << ", iter: " << iterations << endl;

				// random selection of point(s)
				pair<dvec3, dquat> chosenPoint = points[rand() % totalNumberOfPoints];

				// solve model
				dvec3 chosenTrans = glm::inverse(chosenPoint.second) * (sphereCenter - chosenPoint.first);

				// find inliers and compute model
				dvec3 transInterp;
				double transError = 0;
				int inlierCount = 0;
				for (vector<pair<dvec3, dquat>>::iterator p = points.begin(); p != points.end(); p++) {
					dvec3 estCenter = p->first + p->second * chosenTrans;
					double dist = glm::distance(sphereCenter, estCenter);
					//cout << dist << " error" << endl; // DGB
					if (dist < config.inlierTolerance) {
						// this is an inlier
						inlierCount++;

						transInterp += glm::inverse(p->second) * (sphereCenter - p->first);
						transError += dist * dist;
					}
				}

				// DBG
				//cout << inlierCount << "@" << config.minInliers << endl;
				if (inlierCount >= config.minInliers) {
					double curError = glm::sqrt(transError / (inlierCount - 1));
					if (curError < translationErrorOut) {
						translationOut = transInterp / (double)inlierCount;
						translationErrorOut = curError;
					}
					break;
				}
			}
		}

		return translationErrorOut != DBL_MAX;
	}

	bool MagicStylus::computePlaneLSFRANSAC(glm::dvec4 &planeOut, double &fittingErrorOut, vector<dvec3> &points, RansacConfig &config) {
		int totalNumberOfPoints = (int)points.size();
		if (totalNumberOfPoints < config.minInliers || totalNumberOfPoints < 3) {
			return false;
		}

		fittingErrorOut = DBL_MAX;

		int runs = config.numberOfRuns;
		while (runs-- > 0) {
			int iterations = config.getMaximumIterations(3);	// require mininum three points to compute a plane
			while (iterations-- > 0) {
				// random selection of point(s)
				int idx1, idx2, idx3;
				idx1 = rand() % totalNumberOfPoints;
				do {
					idx2 = rand() % totalNumberOfPoints;
				} while (idx2 == idx1);
				do {
					idx3 = rand() % totalNumberOfPoints;
				} while (idx3 == idx1 || idx3 == idx2);

				// solve model
				dvec4 plane = threePointsToPlane(points[idx1], points[idx2], points[idx3]);

				// find inliers
				vector<dvec3> inliers;
				for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
					dvec4 p4(*p, 1);
					double pointPlaneDist = glm::abs(glm::dot(plane, p4));

					if (pointPlaneDist < config.inlierTolerance) {
						inliers.push_back(*p);
					}
				}

				if (inliers.size() >= config.minInliers) {
					glm::dvec4 curPlane;
					double curError = leastSquaresFittingPlane(curPlane, inliers);
					if (curError < fittingErrorOut) {
						planeOut = curPlane;
						fittingErrorOut = curError;
					}
					break;
				}
			}
		}

		return fittingErrorOut != DBL_MAX;
	}

	bool MagicStylus::computeSphereLSFRANSAC(glm::dvec4 &sphereOut, double &fittingErrorOut,
		vector<pair<glm::dvec3, glm::dquat>> &points, RansacConfig &config, int maxIters // maximum number of iterations for least squares fitting
		) {
		int totalNumberOfPoints = (int)points.size();
		if (totalNumberOfPoints < config.minInliers || totalNumberOfPoints < 4) {
			return false;
		}

		fittingErrorOut = DBL_MAX;

		int runs = config.numberOfRuns;
		while (runs-- > 0) {
			int iterations = config.getMaximumIterations(4);	// require mininum four points to compute a sphere
			while (iterations-- > 0) {
				// random selection of point(s)
				int idx1, idx2, idx3, idx4;
				idx1 = rand() % totalNumberOfPoints;
				do {
					idx2 = rand() % totalNumberOfPoints;
				} while (idx2 == idx1);
				do {
					idx3 = rand() % totalNumberOfPoints;
				} while (idx3 == idx1 || idx3 == idx2);
				do {
					idx4 = rand() % totalNumberOfPoints;
				} while (idx4 == idx1 || idx4 == idx2 || idx4 == idx3);

				// solve model
				dvec4 sphere = fourPointsToSphere(points[idx1].first, points[idx2].first, points[idx3].first, points[idx4].first);

				// find inliers
				dvec3 sphereCenter(sphere.x, sphere.y, sphere.z);
				vector<dvec3> inliers;
				for (vector<pair<dvec3, dquat>>::iterator p = points.begin(); p != points.end(); p++) {
					if (glm::abs(glm::distance(p->first, sphereCenter) - sphere.w) < config.inlierTolerance) {
						// this is an inlier
						// cout << "inlier: " << glm::abs(glm::distance(p->first, sphereCenter) - sphere.w) << endl; // DBG
						inliers.push_back(p->first);
					}
				}

				if (inliers.size() >= config.minInliers) {
					glm::dvec4 curSphere;
					double curError = leastSquaresFittingSphere(curSphere, maxIters, inliers);
					if (curError < fittingErrorOut) {
						sphereOut = curSphere;
						fittingErrorOut = curError;
					}
					break;
				}
			}
		}

		return fittingErrorOut != DBL_MAX;
	}

	double MagicStylus::leastSquaresFittingPlane(glm::dvec4 &planeOut, vector<glm::dvec3> &points) {
		// http://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
		int numPoints = (int)points.size();

		// 1. de-mean the points to avoid ill-condition equation
		double xMean = 0, yMean = 0, zMean = 0;
		for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
			xMean += p->x;
			yMean += p->y;
			zMean += p->z;
		}
		xMean /= numPoints;
		yMean /= numPoints;
		zMean /= numPoints;

		vector<dvec3> demeaned;
		for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
			demeaned.push_back(dvec3(p->x - xMean, p->y - yMean, p->z - zMean));
		}

		// 2. construct the equation
		double
			sumX = 0, sumXX = 0, sumXY = 0, sumXZ = 0,
			sumY = 0, sumYY = 0, sumYZ = 0,
			sumZ = 0, sumOne = numPoints;
		for (vector<dvec3>::iterator p = demeaned.begin(); p != demeaned.end(); p++) {
			sumX += p->x;
			sumXX += p->x * p->x;
			sumXY += p->x * p->y;
			sumXZ += p->x * p->z;
			sumY += p->y;
			sumYY += p->y * p->y;
			sumYZ += p->y * p->z;
			sumZ += p->z;
		}


		dmat3x3 J(
			sumXX, sumXY, sumX,
			sumXY, sumYY, sumY,
			sumX, sumY, sumOne
			);
		dvec3 res(sumXZ, sumYZ, sumZ);
		dvec3 coeff = glm::inverse(J) * res;

		// the fitted plane is of the form z - mean(z) = A(x - mean(x)) + B(y - mean(y))
		// so need to rewrite it to the form of Ax + By + Cz + d = 0
		// i.e., Ax + By - z + (mean(z) - A * mean(X) - B * mean(Y)) = 0
		// and we wish to normalize the equation
		double length = glm::length(dvec3(coeff.x, coeff.y, -1));

		planeOut.x = coeff.x / length;
		planeOut.y = coeff.y / length;
		planeOut.z = -1 / length;
		planeOut.w = (zMean - coeff.x * xMean - coeff.y * yMean) / length;

		double error = 0;
		for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
			double diff = planeOut.x * p->x + planeOut.y * p->y + planeOut.z * p->z + planeOut.w;
			error += diff * diff;
		}
		return glm::sqrt(error / numPoints);
	}

	double MagicStylus::leastSquaresFittingSphere(dvec4 &sphereOut, int maxIterations, vector<dvec3> &points) {
		// http://www.geometrictools.com/Documentation/LeastSquaresFitting.pdf
		int numPoints = (int)points.size();

		double xMean = 0, yMean = 0, zMean = 0;
		for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
			xMean += p->x;
			yMean += p->y;
			zMean += p->z;
		}
		xMean /= numPoints;
		yMean /= numPoints;
		zMean /= numPoints;

		double a = xMean, b = yMean, c = zMean;
		double LMean;
		while (maxIterations-- > 0) {
			double prevA = a, prevB = b, prevC = c;
			LMean = 0;
			double LaMean = 0, LbMean = 0, LcMean = 0;
			for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
				double a_minus_x = a - p->x;
				double b_minus_y = b - p->y;
				double c_minus_z = c - p->z;

				double L = glm::sqrt(a_minus_x * a_minus_x + b_minus_y * b_minus_y + c_minus_z * c_minus_z);
				double invL = 1 / L;

				LMean += L;
				LaMean += a_minus_x * invL;
				LbMean += b_minus_y * invL;
				LcMean += c_minus_z * invL;
			}
			LMean /= numPoints;
			LaMean /= numPoints;
			LbMean /= numPoints;
			LcMean /= numPoints;

			a = xMean + LMean * LaMean;
			b = yMean + LMean * LbMean;
			c = zMean + LMean * LcMean;

			if (glm::distance(dvec3(a, b, c), dvec3(prevA, prevB, prevC)) <= DBL_EPSILON) {
				//cout << "exit due to small delta." << endl;
				break;
			}
		}
		sphereOut.x = a;
		sphereOut.y = b;
		sphereOut.z = c;
		sphereOut.w = LMean;

		double error = 0;
		dvec3 sphereCenter(sphereOut.x, sphereOut.y, sphereOut.z);
		for (vector<dvec3>::iterator p = points.begin(); p != points.end(); p++) {
			double diff = glm::distance(*p, sphereCenter) - sphereOut.w;
			//cout << "diff: " << diff << endl; // DBG

			error += diff * diff;
		}
		return glm::sqrt(error / numPoints);
	}

	void GeometryExtendedMarker::resetTransformations() {
		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				transformations[i][j].reset();
			}
		}
		calibrated = false;
	}

	bool GeometryExtendedMarker::calibrateTransformations(vector<vector<Marker>> &markersList) {
		resetTransformations();
		if (markersList.empty()) {
			return false;
		}

		// 0. extract relative poses
		map<int, map<int, vector<tuple<dvec3, dquat, dvec3, dquat>>>> relativePoses = extractRelativePoses(markersList);

		// 1. compute direct transformation using RANSAC
		//RansacConfig translationConfig(
		//	0.999,
		//	0.4,
		//	0.001, // mm
		//	0,	// variable
		//	2000
		//	);
		//RansacConfig rotationConfig(
		//	0.999,
		//	0.4,
		//	1, // degree
		//	0, // variable
		//	2000
		//	);

		for (int i = 0; i < numSubmarkers; i++) {
			// self transformation should be the identity
			transformations[i][i].direct = transformations[i][i].valid = true;

			for (int j = i + 1; j < numSubmarkers; j++) {
				int numSamplePairs = (int)relativePoses.at(i).at(j).size();
				//if (numSamplePairs > translationConfig.minInliers && numSamplePairs > rotationConfig.minInliers) {
				cout << "Calibrating " << i << " and " << j << " with " << numSamplePairs << " sample pairs." << endl;

				dvec3 translation;
				dquat rotation;
				double error;

				bool i2jTransValid, i2jRotValid, j2iTransValid, j2iRotValid;
				i2jTransValid = i2jRotValid = j2iTransValid = j2iRotValid = false;

				defaultTranslationRansacConfig.minInliers = (int)glm::floor(relativePoses.at(i).at(j).size() * (1 - defaultTranslationRansacConfig.outlierProbability));
				defaultRotationRansacConfig.minInliers = (int)glm::floor(relativePoses.at(i).at(j).size() * (1 - defaultRotationRansacConfig.outlierProbability));

				if (computeTranslationRANSAC(translation, error, relativePoses.at(i).at(j), defaultTranslationRansacConfig, false)) {
					transformations[i][j].translation = translation;
					transformations[i][j].translationError = error;
					i2jTransValid = true;

					// DBG
					cout << "Translation " << i << "->" << j << " succeeded: (" << translation.x << "," << translation.y << "," << translation.z << ") with error " << error << endl;
				}

				if (computeTranslationRANSAC(translation, error, relativePoses.at(i).at(j), defaultTranslationRansacConfig, true)) {
					transformations[j][i].translation = translation;
					transformations[j][i].translationError = error;
					j2iTransValid = true;

					// DBG
					cout << "Translation " << j << "->" << i << " succeeded: (" << translation.x << "," << translation.y << "," << translation.z << ") with error " << error << endl;
				}

				if (computeRotationRANSAC(rotation, error, relativePoses.at(i).at(j), defaultRotationRansacConfig)) {
					transformations[i][j].rotation = rotation;
					transformations[i][j].rotationError = error;
					i2jRotValid = true;

					transformations[j][i].rotation = glm::inverse(rotation);
					transformations[j][i].rotationError = error;
					j2iRotValid = true;

					// DBG
					dvec3 eAngles = glm::eulerAngles(rotation) * 180.0 / pi<double>();
					cout << "Rotation " << i << "->" << j << " succeeded: (" << eAngles.x << "," << eAngles.y << "," << eAngles.z << ") with error " << error << endl;
				}

				transformations[i][j].direct = transformations[i][j].valid = i2jTransValid && i2jRotValid;
				transformations[j][i].direct = transformations[j][i].valid = j2iTransValid && j2iRotValid;
				//}
			}
		}

		// 2. compute indirect transformation using the Floyd–Warshall algorithm
		for (int k = 0; k < numSubmarkers; k++) {
			for (int i = 0; i < numSubmarkers; i++) {
				for (int j = 0; j < numSubmarkers; j++) {
					if (transformations[i][k].valid && transformations[k][j].valid) {
						// we update rotation and translation independently
						double combinedRotationError =
							transformations[i][k].rotationError + transformations[k][j].rotationError;
						if (!transformations[i][j].valid || combinedRotationError < transformations[i][j].rotationError) {
							transformations[i][j].rotationError = combinedRotationError;
							transformations[i][j].rotation = transformations[i][k].rotation * transformations[k][j].rotation;
						}

						double combinedTranslationError =
							transformations[i][k].translationError + transformations[k][j].translationError;
						if (!transformations[i][j].valid || combinedTranslationError < transformations[i][j].translationError) {
							transformations[i][j].translationError = combinedTranslationError;
							transformations[i][j].translation = transformations[i][k].translation + transformations[i][k].rotation * transformations[k][j].translation;
						}

						transformations[i][j].valid = true;
					}
				}
			}
		}

		//// 3. check if we have a fully connected graph
		//for (int i = 0; i < numSubmarkers; i++) {
		//	for (int j = 0; j < numSubmarkers; j++) {
		//		if (!transformations[i][j].valid) {
		//			cout << "Missing transformation between submarker " << i << " and submarker " << j << endl;
		//			return false;
		//		}
		//	}
		//}

		//// 4. find three submarkers for determining the global orientation
		//double var = DBL_MAX;
		//for (int a = 0; a < numSubmarkers; a++) {
		//	for (int b = 0; b < numSubmarkers; b++) {
		//		for (int c = 0; c < numSubmarkers; c++) {
		//			if (a == b || a == c || b == c) continue;

		//			dvec3 ab = transformations.at(a).at(b).translation;
		//			dvec3 ac = transformations.at(a).at(c).translation;
		//			double angleA = glm::angle(normalize(ab), normalize(ac));

		//			dvec3 ba = transformations.at(b).at(a).translation;
		//			dvec3 bc = transformations.at(b).at(c).translation;
		//			double angleB = glm::angle(normalize(ba), normalize(bc));

		//			dvec3 ca = transformations.at(c).at(a).translation;
		//			dvec3 cb = transformations.at(c).at(b).translation;
		//			double angleC = glm::angle(normalize(ca), normalize(cb));

		//			double meanAngle = (angleA + angleB + angleC) / 3;
		//			double tmpVar =
		//				(angleA - meanAngle) * (angleA - meanAngle) +
		//				(angleB - meanAngle) * (angleB - meanAngle) +
		//				(angleC - meanAngle) * (angleC - meanAngle);

		//			if (tmpVar < var) {
		//				var = tmpVar;
		//				orientationPointA = a;
		//				orientationPointB = b;
		//				orientationPointC = c;
		//				orientationPlaneNormal = normalize(cross(ab, ac));
		//			}
		//		}
		//	}
		//}

		calibrated = true;
		return calibrated;
	}

	bool GeometryExtendedMarker::isCalibrated() {
		return calibrated;
	}

	double GeometryExtendedMarker::getMaxTranslationError() {
		double maxTransErr = DBL_MIN;
		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				if (transformations[i][j].valid && transformations[i][j].translationError > maxTransErr) {
					maxTransErr = transformations[i][j].translationError;
				}
			}
		}
		return maxTransErr;
	}
	double GeometryExtendedMarker::getMaxRotationError() {
		double maxRotErr = DBL_MIN;
		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				if (transformations[i][j].valid && transformations[i][j].rotationError > maxRotErr) {
					maxRotErr = transformations[i][j].rotationError;
				}
			}
		}
		return maxRotErr;
	}

	void GeometryExtendedMarker::printTransformationSummary() {
		int directCount = 0;
		int indirectCount = 0;
		int validCount = 0;
		double maxTransErr = 0, maxRotErr = 0;

		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				if (transformations[i][j].valid) {
					validCount++;

					if (i != j) {
						if (transformations[i][j].direct) {
							directCount++;
						}
						else {
							indirectCount++;
						}
					}

					if (transformations[i][j].translationError > maxTransErr) {
						maxTransErr = transformations[i][j].translationError;
					}

					if (transformations[i][j].rotationError > maxRotErr) {
						maxRotErr = transformations[i][j].rotationError;
					}
				}
			}
		}

		cout << "Calibraiton summary:" << endl;
		cout << "\t " << validCount << " out of " << (numSubmarkers * numSubmarkers) << " pairs calibrated." << endl;
		cout << "\t " << numSubmarkers << " are self-transformation." << endl;
		cout << "\t " << directCount << " are direct transformation." << endl;
		cout << "\t " << indirectCount << " are indirect transformation." << endl;
		cout << "\t Maximum translation error is " << maxTransErr << " and maximum rotation error is " << maxRotErr << "." << endl;
		cout << "\t Base marker is " << baseSubmarker << "." << endl;

		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				if (transformations[i][j].valid) {
					dvec3 eAngles = glm::eulerAngles(transformations[i][j].rotation) * 180.0 / pi<double>();
					cout << i << "->" << j << ":"
						<< " t(" << transformations[i][j].translation.x << "," << transformations[i][j].translation.y << "," << transformations[i][j].translation.z << ")"
						<< " r(" << eAngles.x << "," << eAngles.y << "," << eAngles.z << ")" << endl;
				}
			}
		}

	}

	void GeometryExtendedMarker::saveCalibrations(string file) {
		ofstream out(file, ios::out);
		out << calibrated << " " << baseSubmarker << endl;
		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				InterMarkerTransformation trans = transformations[i][j];
				out
					<< trans.valid << " " << trans.direct
					<< " " << trans.translation.x << " " << trans.translation.y << " " << trans.translation.z
					<< " " << trans.translationError
					<< " " << trans.rotation.x << " " << trans.rotation.y << " " << trans.rotation.z << " " << trans.rotation.w
					<< " " << trans.rotationError
					<< endl;
			}
		}
		out.close();
	}

	void MagicStylus::saveCalibrations(string file) {
		ofstream out(file, ios::out);
		out
			<< stylusTipTransformation.valid << " "
			<< stylusTipTransformation.translation.x << " "
			<< stylusTipTransformation.translation.y << " "
			<< stylusTipTransformation.translation.z << " "
			<< stylusTipTransformation.translationError
			<< endl;
		out.close();
	}

	void GeometryExtendedMarker::loadCalibrations(string file) {
		resetTransformations();
		ifstream in(file, ios::in);
		in >> calibrated;
		in >> baseSubmarker;
		for (int i = 0; i < numSubmarkers; i++) {
			for (int j = 0; j < numSubmarkers; j++) {
				in >> transformations[i][j].valid;
				in >> transformations[i][j].direct;
				in >> transformations[i][j].translation.x;
				in >> transformations[i][j].translation.y;
				in >> transformations[i][j].translation.z;
				in >> transformations[i][j].translationError;
				in >> transformations[i][j].rotation.x;
				in >> transformations[i][j].rotation.y;
				in >> transformations[i][j].rotation.z;
				in >> transformations[i][j].rotation.w;
				in >> transformations[i][j].rotationError;
			}
		}
		in.close();
	}

	void MagicStylus::loadCalibrations(string file) {
		resetStylusTipTransformation();
		ifstream in(file, ios::in);
		in >> stylusTipTransformation.valid;
		in >> stylusTipTransformation.translation.x;
		in >> stylusTipTransformation.translation.y;
		in >> stylusTipTransformation.translation.z;
		in >> stylusTipTransformation.translationError;
		in.close();
	}

	bool GeometryExtendedMarker::estimateSubmarkerPose(dvec3 &positionOut, dquat &orientationOut, int id, map<int, pair<dvec3, dquat>> &detectedPoses) {
		double totalTransErr = 0, totalRotErr = 0;
		for (map<int, pair<dvec3, dquat>>::iterator p = detectedPoses.begin(); p != detectedPoses.end(); p++) {
			if (transformations[p->first][id].valid) {
				totalTransErr += 1.0 / transformations[p->first][id].translationError;
				totalRotErr += 1.0 / transformations[p->first][id].rotationError;
			}
		}
		dvec3 posInterp;
		dquat oriInterp;
		double sumOriWeight = 0;

		for (map<int, pair<dvec3, dquat>>::iterator p = detectedPoses.begin(); p != detectedPoses.end(); p++) {
			if (p->first != id && transformations[p->first][id].valid) {
				dvec3 estPos = p->second.first + p->second.second * transformations[p->first][id].translation;
				dquat estOri = p->second.second * transformations[p->first][id].rotation;

				double normalisedPosWeight = 1.0 / (transformations[p->first][id].translationError * totalTransErr);
				double normalisedOriWeight = 1.0 / (transformations[p->first][id].rotationError * totalRotErr);

				posInterp += estPos * normalisedPosWeight;
				if (sumOriWeight == 0) {
					sumOriWeight = normalisedOriWeight;
					oriInterp = estOri;
				}
				else {
					sumOriWeight += normalisedOriWeight;
					oriInterp = glm::slerp(estOri, oriInterp, normalisedOriWeight / sumOriWeight);
				}
			}
		}

		bool hasEstimatedPose = sumOriWeight > 0;
		bool hasDetectedPose = false;

		if (detectedPoses.find(id) != detectedPoses.end()) {
			hasDetectedPose = true;
			if (hasEstimatedPose) {
				posInterp = (posInterp + detectedPoses.at(id).first) / 2.0;
				oriInterp = glm::slerp(detectedPoses.at(id).second, oriInterp, 0.5); // i.e., simple averaging
			}
			else {
				posInterp = detectedPoses.at(id).first;
				oriInterp = detectedPoses.at(id).second;
			}
		}

		if (hasDetectedPose || hasEstimatedPose) {
			positionOut = posInterp;
			orientationOut = oriInterp;
			return true;
		}
		else {
			return false;
		}
	}

	map<int, pair<dvec3, dquat>> GeometryExtendedMarker::estimateSubmarkerPoses(vector<Marker> &allMarkers) {
		map<int, pair<dvec3, dquat>> detectedPoses;
		if (!calibrated || allMarkers.empty()) {
			return detectedPoses;
		}

		vector<Marker> markers = filterOutliers(allMarkers);

		for (auto m = markers.begin(); m != markers.end(); m++) {
			detectedPoses.insert(make_pair(m->id, getMarkerPose(*m)));
		}

		/*for (int i = 0; i < numSubmarkers; i++) {
			if (detectedPoses.find(i) != detectedPoses.end()) {
				poseSmoother[i].addNewPose(detectedPoses.at(i).first, detectedPoses.at(i).second);
				auto newPose = poseSmoother[i].getAveragePose();
				if (get<0>(newPose)) {
					detectedPoses.at(i).first = get<1>(newPose);
					detectedPoses.at(i).second = get<2>(newPose);
				}
			}
			else {
				poseSmoother[i].addNullPose();
				auto newPose = poseSmoother[i].getAveragePose();
				if (get<0>(newPose)) {
					detectedPoses.insert(make_pair(i, make_pair(get<1>(newPose), get<2>(newPose))));
				}
			}
		}*/

		map<int, pair<dvec3, dquat>> estimatedPoses;
		for (int i = 0; i < numSubmarkers; i++) {
			dvec3 pos;
			dquat ori;
			if (estimateSubmarkerPose(pos, ori, i, detectedPoses)) {
				estimatedPoses.insert(pair<int, pair<dvec3, dquat>>(i, pair<dvec3, dquat>(pos, ori)));
			}
		}
		return estimatedPoses;
	}

	bool GeometryExtendedMarker::estimateGlobalPoseUsingBaseSubmarker(dvec3 &pos, dquat &ori, map<int, pair<dvec3, dquat>> &submarkerPoses) {
		if (submarkerPoses.find(baseSubmarker) != submarkerPoses.end()) {
			pos = submarkerPoses.at(baseSubmarker).first;
			ori = submarkerPoses.at(baseSubmarker).second;
			return true;
		}
		return false;
	}

	bool GeometryExtendedMarker::estimateGlobalPose(glm::dvec3 &posOut, glm::dquat &oriOut, map<int, pair<glm::dvec3, glm::dquat>> &submarkerPoses) {
		return estimateGlobalPoseUsingBaseSubmarker(posOut, oriOut, submarkerPoses);
	}

	void MagicStylus::resetStylusTipTransformation() {
		stylusTipTransformation.reset();
	}

	void MagicStylus::resetDrawingPlane() {
		drawingPlane = dvec4();
	}

	bool MagicStylus::calibrateDrawingPlane(vector<dvec3> points) {
		resetDrawingPlane();
		//RansacConfig planeConfig(0.999, 0.2, 0.002, (int)glm::floor(points.size() * 0.7), 1000);
		defaultPlaneFittingRansacConfig.minInliers = (int)glm::floor(points.size() * (1 - defaultPlaneFittingRansacConfig.outlierProbability));

		dvec4 plane;
		double error;

		if (computePlaneLSFRANSAC(plane, error, points, defaultPlaneFittingRansacConfig)) {
			// DBG
			cout << "Plane: " << plane.x << "x + " << plane.y << "y + " << plane.z << "z + " << plane.w << " = 0 @ error = " << error << endl;

			drawingPlane = plane;
			return true;
		}
		else {
			return false;
		}
	}

	bool MagicStylus::onPlane(dvec3 &point) {
		dvec4 p4(point, 1);
		return hasDrawingPlane() && glm::abs(glm::dot(drawingPlane, p4)) < planeDistMu;
	}

	void MagicStylus::setPlaneDistanceThreshold(double mu) {
		if (mu > 0) {
			planeDistMu = mu;
		}
	}

	double MagicStylus::getPlaneDistanceThreshold() {
		return planeDistMu;
	}

	dvec4 MagicStylus::getDrawingPlane() {
		return drawingPlane;
	}

	bool MagicStylus::hasDrawingPlane() {
		return glm::distance(dvec4(), drawingPlane) > 0;
	}

	bool MagicStylus::calibrateStylusTipTransformation(vector<pair<glm::dvec3, glm::dquat>> &posesList) {
		resetStylusTipTransformation();

		//RansacConfig sphereConfig(0.999, 0.2, 0.002, (int)glm::floor(posesList.size() * 0.7), 2000);
		defaultSphereFittingRansacConfig.minInliers = (int)glm::floor(posesList.size() * (1 - defaultSphereFittingRansacConfig.outlierProbability));
		int maxFittingIterations = 1000;

		//cout << sphereConfig.getMaximumIterations(4) << " iterations. " << endl;

		dvec4 sphere;
		double error;
		if (computeSphereLSFRANSAC(sphere, error, posesList, defaultSphereFittingRansacConfig, maxFittingIterations)) {
			// DBG
			cout << "Sphere: (x - " << sphere.x << ")^2 + (y - " << sphere.y << ")^2 + (z - " << sphere.z << ")^2 = " << sphere.w << "^2 @ error = " << error << endl;

			//RansacConfig translationConfig(0.999, 0.2, 0.005, (int)glm::floor(posesList.size() * 0.50), 2000);
			defaultTipTranslationRansacConfig.minInliers = (int)glm::floor(posesList.size() * (1 - defaultTipTranslationRansacConfig.outlierProbability));
			dvec3 translation;
			if (computeStylusTipTranslationRANSAC(translation, error, sphere, posesList, defaultTipTranslationRansacConfig)) {
				stylusTipTransformation.translation = translation;
				stylusTipTransformation.translationError = error;
				stylusTipTransformation.valid = true;

				// DBG
				cout << "Stylus tip: t(" << stylusTipTransformation.translation.x << "," << stylusTipTransformation.translation.y << "," << stylusTipTransformation.translation.z << ")@"
					<< stylusTipTransformation.translationError << endl;
			}
		}

		return stylusTipTransformation.valid;
	}

	bool MagicStylus::getStylusTipPosition(dvec3 &posOut, dvec3 &basePos, dquat &baseOri) {
		if (stylusTipTransformation.valid) {
			posOut = basePos + baseOri * stylusTipTransformation.translation;
			return true;
		}
		else {
			return false;
		}
	}

	MagicStylus::MagicStylus() :
		defaultPlaneFittingRansacConfig(0.999, 0.3, 0.002, 0, 1000),
		defaultSphereFittingRansacConfig(0.999, 0.3, 0.002, 0, 2000),
		defaultTipTranslationRansacConfig(0.999, 0.3, 0.005, 0, 2000),
		planeDistMu(0.001)
	{
	}
	MagicStylus::~MagicStylus(){}

	LinearKalmanFilterTracker::~LinearKalmanFilterTracker() {}
	LinearKalmanFilterTracker::LinearKalmanFilterTracker() {
		measurementMatrix.setZero();
		measurementMatrix(0, 0) = 1;
		measurementMatrix(1, 1) = 1;
		measurementMatrix(2, 2) = 1;
		measurementMatrix(3, 9) = 1;
		measurementMatrix(4, 10) = 1;
		measurementMatrix(5, 11) = 1;

		measurementMatrixTranspose = measurementMatrix.transpose();
	}

	void LinearKalmanFilterTracker::reset(Matrix<double, 18, 1> &processNoise, Matrix<double, 6, 1> &measurementNoise, dvec3 &initialPosition, dquat &initialOrientation) {
		dvec3 angles = glm::eulerAngles(initialOrientation);
		statePost <<
			initialPosition.x, initialPosition.y, initialPosition.z, 0, 0, 0, 0, 0, 0, // i.e., zero velocity and acceleration
			angles.x, angles.y, angles.z, 0, 0, 0, 0, 0, 0; // i.e., zero angular velocity and angular acceleration

		processNoiseCov.setIdentity();
		for (int i = 0; i < 18; i++) {
			processNoiseCov(i, i) = processNoise(i);
		}

		measurementNoiseCov.setIdentity();
		for (int i = 0; i < 6; i++) {
			measurementNoiseCov(i, i) = measurementNoise(i);
		}

		errorCovPost.setIdentity();
	}

	void LinearKalmanFilterTracker::predictAndCorrect(dvec3 &measuredPosition, dquat &measuredOrientation, double deltaT) {
		MatrixXd stateOld = statePost;

		double halfDeltaTSquare = 0.5 * deltaT * deltaT;
		transitionMatrix.setIdentity();
		transitionMatrix(0, 3) = deltaT; transitionMatrix(0, 6) = halfDeltaTSquare;
		transitionMatrix(1, 4) = deltaT; transitionMatrix(1, 7) = halfDeltaTSquare;
		transitionMatrix(2, 5) = deltaT; transitionMatrix(2, 8) = halfDeltaTSquare;
		transitionMatrix(3, 6) = deltaT;
		transitionMatrix(4, 7) = deltaT;
		transitionMatrix(5, 8) = deltaT;
		transitionMatrix(9, 12) = deltaT; transitionMatrix(9, 15) = halfDeltaTSquare;
		transitionMatrix(10, 13) = deltaT; transitionMatrix(10, 16) = halfDeltaTSquare;
		transitionMatrix(11, 14) = deltaT; transitionMatrix(11, 17) = halfDeltaTSquare;
		transitionMatrix(12, 15) = deltaT;
		transitionMatrix(13, 16) = deltaT;
		transitionMatrix(14, 17) = deltaT;

		Matrix<double, 6, 1> measurement;
		dvec3 measuredAngles = glm::eulerAngles(measuredOrientation);

		measurement <<
			measuredPosition.x, measuredPosition.y, measuredPosition.z,
			measuredAngles.x, measuredAngles.y, measuredAngles.z;

		// predict
		MatrixXd statePre = transitionMatrix * statePost;
		MatrixXd errorCovPre = transitionMatrix * errorCovPost * transitionMatrix.transpose() + processNoiseCov;

		// correct
		MatrixXd innovation = measurement - measurementMatrix * statePre;
		MatrixXd innovationCov = measurementMatrix * errorCovPre * measurementMatrixTranspose + measurementNoiseCov;
		MatrixXd kalmanGain = errorCovPre * measurementMatrixTranspose * innovationCov.inverse();

		statePost = statePre + kalmanGain * innovation;
		errorCovPost = (Matrix<double, 18, 18>::Identity() - kalmanGain * measurementMatrix) * errorCovPre;
	}

	pair<dvec3, dquat> LinearKalmanFilterTracker::getEstimatedPose() {
		dvec3 pos(statePost(0), statePost(1), statePost(2));
		dquat ori(dvec3(statePost(9), statePost(10), statePost(11)));

		return pair<dvec3, dquat>(pos, ori);
	}

	pair<dvec3, dquat> LinearKalmanFilterTracker::getGhostPose(double deltaT) {
		double halfDeltaTSquare = 0.5 * deltaT * deltaT;
		transitionMatrix.setIdentity();
		transitionMatrix(0, 3) = deltaT; transitionMatrix(0, 6) = halfDeltaTSquare;
		transitionMatrix(1, 4) = deltaT; transitionMatrix(1, 7) = halfDeltaTSquare;
		transitionMatrix(2, 5) = deltaT; transitionMatrix(2, 8) = halfDeltaTSquare;
		transitionMatrix(3, 6) = deltaT;
		transitionMatrix(4, 7) = deltaT;
		transitionMatrix(5, 8) = deltaT;
		transitionMatrix(9, 12) = deltaT; transitionMatrix(9, 15) = halfDeltaTSquare;
		transitionMatrix(10, 13) = deltaT; transitionMatrix(10, 16) = halfDeltaTSquare;
		transitionMatrix(11, 14) = deltaT; transitionMatrix(11, 17) = halfDeltaTSquare;
		transitionMatrix(12, 15) = deltaT;
		transitionMatrix(13, 16) = deltaT;
		transitionMatrix(14, 17) = deltaT;

		MatrixXd ghostState = transitionMatrix * statePost;

		dvec3 pos(ghostState(0), ghostState(1), ghostState(2));
		dquat ori(dvec3(ghostState(9), ghostState(10), ghostState(11)));

		//cout << "Ghost DeltaT: " << deltaT << endl;	// DBG
		//cout << ghostState.topRows(3) << endl;	// DBG

		return pair<dvec3, dquat>(pos, ori);
	}

	PositionTracker::~PositionTracker() {}
	PositionTracker::PositionTracker() {
		measurementMatrix.setZero();
		measurementMatrix(0, 0) = 1;
		measurementMatrix(1, 1) = 1;
		measurementMatrix(2, 2) = 1;

		measurementMatrixTranspose = measurementMatrix.transpose();
	}

	void PositionTracker::reset(Matrix<double, 9, 1> &processNoise, Matrix<double, 3, 1> &measurementNoise, dvec3 &initialPosition) {
		statePost <<
			initialPosition.x, initialPosition.y, initialPosition.z, 0, 0, 0, 0, 0, 0; // i.e., zero velocity and acceleration

		processNoiseCov.setZero();
		for (int i = 0; i < 9; i++) {
			processNoiseCov(i, i) = processNoise(i) * processNoise(i);
		}

		measurementNoiseCov.setZero();
		for (int i = 0; i < 3; i++) {
			measurementNoiseCov(i, i) = measurementNoise(i) * measurementNoise(i);
		}

		errorCovPost.setIdentity();
	}

	void PositionTracker::predictAndCorrect(dvec3 &measuredPosition, double deltaT) {
		double halfDeltaTSquare = 0.5 * deltaT * deltaT;
		transitionMatrix.setIdentity();
		transitionMatrix(0, 3) = deltaT; transitionMatrix(0, 6) = halfDeltaTSquare;
		transitionMatrix(1, 4) = deltaT; transitionMatrix(1, 7) = halfDeltaTSquare;
		transitionMatrix(2, 5) = deltaT; transitionMatrix(2, 8) = halfDeltaTSquare;
		transitionMatrix(3, 6) = deltaT;
		transitionMatrix(4, 7) = deltaT;
		transitionMatrix(5, 8) = deltaT;

		Matrix<double, 3, 1> measurement;

		measurement <<
			measuredPosition.x, measuredPosition.y, measuredPosition.z;

		// predict
		MatrixXd statePre = transitionMatrix * statePost;
		MatrixXd errorCovPre = transitionMatrix * errorCovPost * transitionMatrix.transpose() + processNoiseCov;

		// correct
		MatrixXd innovation = measurement - measurementMatrix * statePre;
		MatrixXd innovationCov = measurementMatrix * errorCovPre * measurementMatrixTranspose + measurementNoiseCov;
		MatrixXd kalmanGain = errorCovPre * measurementMatrixTranspose * innovationCov.inverse();

		statePost = statePre + kalmanGain * innovation;
		errorCovPost = (Matrix<double, 9, 9>::Identity() - kalmanGain * measurementMatrix) * errorCovPre;
	}

	dvec3 PositionTracker::getEstimatedPosition() {
		return dvec3(statePost(0), statePost(1), statePost(2));
	}

	dvec3 PositionTracker::getGhostPosition(double deltaT) {
		double halfDeltaTSquare = 0.5 * deltaT * deltaT;
		transitionMatrix.setIdentity();
		transitionMatrix(0, 3) = deltaT; transitionMatrix(0, 6) = halfDeltaTSquare;
		transitionMatrix(1, 4) = deltaT; transitionMatrix(1, 7) = halfDeltaTSquare;
		transitionMatrix(2, 5) = deltaT; transitionMatrix(2, 8) = halfDeltaTSquare;
		transitionMatrix(3, 6) = deltaT;
		transitionMatrix(4, 7) = deltaT;
		transitionMatrix(5, 8) = deltaT;

		MatrixXd ghostState = transitionMatrix * statePost;

		return dvec3(ghostState(0), ghostState(1), ghostState(2));
	}
}
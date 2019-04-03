#pragma once

#include <unordered_map>
#include <map>
#include <tuple>
#include <deque>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <aruco/aruco.h>
//#include <aruco/highlyreliablemarkers.h>

#include <Eigen/Dense>

using namespace std;

namespace gem {
	glm::dmat4 glMatFromVecQuat(glm::dvec3 pos, glm::dquat ori);
	pair<glm::dvec3, glm::dquat> getMarkerPose(aruco::Marker &marker);

	class RansacConfig {
	public:
		double desiredRobustness; //!< probability of getting a good sample (without outlier)
		double outlierProbability; //!< probablity of observation an outlier
		double inlierTolerance; //!< maximum distance from the model to be considered as inlier
		int numberOfRuns; //!< number of runs
		int minInliers;	//!< minimum number of inliers for fitting the final model

		RansacConfig(double pi, double rho, double mu, int tau, int theta) :
			desiredRobustness(pi),
			outlierProbability(rho),
			inlierTolerance(mu),
			minInliers(tau),
			numberOfRuns(theta)
		{
		}

		/**
		 * This computes the maximum number of attempts needed for RANSAC
		 * https://en.wikipedia.org/wiki/RANSAC#Parameters
		 */
		inline int getMaximumIterations(int minFittingPoints) {
			return (int)glm::ceil(glm::log(1 - desiredRobustness) / glm::log(1 - glm::pow((1 - outlierProbability), minFittingPoints)));
		}
	};

	class PoseRunningAverager {
	private:
		deque<tuple<bool, glm::dvec3, glm::dquat>> poses;
		int numPoses;
		int maxPoses;
		void removeOldPose();
	public:
		PoseRunningAverager(int mP);
		~PoseRunningAverager();

		void addNewPose(glm::dvec3, glm::dquat);
		void addNullPose();
		tuple<bool, glm::dvec3, glm::dquat> getAveragePose();
	};

	class InterMarkerTransformation {
	public:
		glm::dvec3 translation;
		glm::dquat rotation;

		double translationError; //!< RMS error from the calibration data set
		double rotationError;	//!< RMS error from the calibration data set

		bool valid;	//!< whether this transformation has been calibrated successfully
		bool direct; //!< calibrated via direct measurement or indirect transformation

		InterMarkerTransformation() :
			translationError(0),
			rotationError(0),
			valid(false),
			direct(false)
		{
		}

		void reset() {
			translation = glm::dvec3();
			rotation = glm::dquat();
			translationError = 0;
			translationError = 0;
			valid = false;
			direct = false;
		}
	};

	/**
	 * A linear kalman filter for tracking the GEM marker global pose.
	 * https://en.wikipedia.org/wiki/Kalman_filter
	 * http://campar.in.tum.de/Chair/KalmanFilter
	 */
	class LinearKalmanFilterTracker {
	public:
		Eigen::Matrix<double, 18, 18> transitionMatrix;
		Eigen::Matrix<double, 6, 18> measurementMatrix;
		Eigen::Matrix<double, 18, 6> measurementMatrixTranspose;
		Eigen::Matrix<double, 18, 18> errorCovPost;
		Eigen::Matrix<double, 18, 18> processNoiseCov;
		Eigen::Matrix<double, 6, 6> measurementNoiseCov;
		/**
		 * State = (x,y,z,\dot{x},\dot{y},\dot{z},\ddot{x},\ddot{y},\ddot{z},psi,theta,phi,\dot{psi},\dot{theta},\dot{phi},\ddot{psi},\ddot{theta},\ddot{phi})
		 */
		Eigen::Matrix<double, 18, 1> statePost;

		LinearKalmanFilterTracker();
		~LinearKalmanFilterTracker();

		void reset(Eigen::Matrix<double, 18, 1> &processNoise, Eigen::Matrix<double, 6, 1> &measurementNoise, glm::dvec3 &initialPosition, glm::dquat &initialOrientation);
		void predictAndCorrect(glm::dvec3 &measuredPosition, glm::dquat &measuredOrientation, double deltaT);
		pair<glm::dvec3, glm::dquat> getEstimatedPose();

		/**
		 * Estimate ghost pose using dead reckoning.
		 */
		pair<glm::dvec3, glm::dquat> getGhostPose(double deltaT);
	};

	/**
	 * Using Kalman Filter for tracking 3D position.
	 */
	class PositionTracker {
	public:
		Eigen::Matrix<double, 9, 9> transitionMatrix;
		Eigen::Matrix<double, 3, 9> measurementMatrix;
		Eigen::Matrix<double, 9, 3> measurementMatrixTranspose;
		Eigen::Matrix<double, 9, 9> errorCovPost;
		Eigen::Matrix<double, 9, 9> processNoiseCov;
		Eigen::Matrix<double, 3, 3> measurementNoiseCov;
		/**
		* State = (x,y,z,\dot{x},\dot{y},\dot{z},\ddot{x},\ddot{y},\ddot{z})
		*/
		Eigen::Matrix<double, 9, 1> statePost;

		PositionTracker();
		~PositionTracker();

		void reset(Eigen::Matrix<double, 9, 1> &processNoise, Eigen::Matrix<double, 3, 1> &measurementNoise, glm::dvec3 &initialPosition);
		void predictAndCorrect(glm::dvec3 &measuredPosition, double deltaT);
		glm::dvec3 getEstimatedPosition();

		/**
		* Estimate ghost position using dead reckoning.
		*/
		glm::dvec3 getGhostPosition(double deltaT);
	};

	class GeometryExtendedMarker {
	private:
		int poseRunningAverageBufferSize;
		//vector<PoseRunningAverager> poseSmoother;

		float submarkerSize; //!< size of each submarker in meters
		aruco::Dictionary dictionary;	//!< the Aruco dictionary that holds information about the submarkers
		int numSubmarkers; //!< number of submarkers (= dictionary.size())

		int baseSubmarker; //!< the submarker used for representing the global pose

		vector<vector<InterMarkerTransformation>> transformations; //!< inter-submarker pose transformations
		bool calibrated; //!< indicates whether the GEM has been calibrated.

		double rangeNear, rangeFar; //!< the near plane and the far plane for tracking. Used for removing false positives
		double distMu; //!< used for deciding whether two detected submarkers are both inliers
		double rotMu; //!< used for deciding whether two detected submarkers are both inliers

		bool computeTranslationRANSAC(glm::dvec3 &translationOut, double &translationErrorOut,
			vector<tuple<glm::dvec3, glm::dquat, glm::dvec3, glm::dquat>> &pairs,
			RansacConfig &config, bool inverse);
		bool computeRotationRANSAC(glm::dquat &rotationOut, double &rotationErrorOut,
			vector<tuple<glm::dvec3, glm::dquat, glm::dvec3, glm::dquat>> &pairs,
			RansacConfig &config);

		bool estimateSubmarkerPose(glm::dvec3 &positionOut, glm::dquat &orientationOut, int id, map<int, pair<glm::dvec3, glm::dquat>> &detectedPoses);
		bool estimateGlobalPoseUsingOrientationPoints(glm::dvec3 &posOut, glm::dquat &oriOut, map<int, pair<glm::dvec3, glm::dquat>> &submarkerPoses);
		bool estimateGlobalPoseUsingBaseSubmarker(glm::dvec3 &posOut, glm::dquat &oriOut, map<int, pair<glm::dvec3, glm::dquat>> &submarkerPoses);
	public:
		enum GlobalPoseMethod {
			USING_BASE_MARKER,
			USING_ORIENTATION_POINTS
		};

		RansacConfig defaultTranslationRansacConfig;	//!< used for inter-submarker translation calibration
		RansacConfig defaultRotationRansacConfig;	//!< used for inter-submarker rotation calibration


		GeometryExtendedMarker(float mSize, aruco::Dictionary dict);
		~GeometryExtendedMarker();

		void setTrackingRange(double near, double far);
		void setTrackingDistanceThreshold(double mu);
		void setTrackingRotationThreshold(double mu);

		map<int, map<int, vector<tuple<glm::dvec3, glm::dquat, glm::dvec3, glm::dquat>>>>
			GeometryExtendedMarker::extractRelativePoses(vector<vector<aruco::Marker>> &calibrationSamples);
		vector<aruco::Marker> filterOutliers(vector<aruco::Marker> &markers);

		const aruco::Dictionary getDictionary();

		bool calibrateTransformations(vector<vector<aruco::Marker>> &markersList);
		void resetTransformations();
		void printTransformationSummary();
		bool isCalibrated();
		void setBaseSubmarker(int id);
		int getBaseSubmarker();
		double getMaxTranslationError();
		double getMaxRotationError();

		map<int, pair<glm::dvec3, glm::dquat>> estimateSubmarkerPoses(vector<aruco::Marker> &markers);
		bool estimateGlobalPose(glm::dvec3 &posOut, glm::dquat &oriOut, map<int, pair<glm::dvec3, glm::dquat>> &submarkerPoses);

		void saveCalibrations(string file);
		void loadCalibrations(string file);
	};

	class MagicStylus {
	private:
		InterMarkerTransformation stylusTipTransformation;
		glm::dvec4 drawingPlane;
		double planeDistMu; //!< distance threshold (in meters) for deciding whether a point is on the plane or not

		glm::dvec4 fourPointsToSphere(glm::dvec3 &p1, glm::dvec3 &p2, glm::dvec3 &p3, glm::dvec3 &p4);
		double leastSquaresFittingSphere(glm::dvec4 &sphereOut, int maxIterations, vector<glm::dvec3> &points);

		glm::dvec4 threePointsToPlane(glm::dvec3 &p1, glm::dvec3 &p2, glm::dvec3 &p3);
		double leastSquaresFittingPlane(glm::dvec4 &planeOut, vector<glm::dvec3> &points);

		bool computeSphereLSFRANSAC(glm::dvec4 &sphereOut, double &fittingErrorOut,
			vector<pair<glm::dvec3, glm::dquat>> &points,
			RansacConfig &config, int maxIters);
		bool computeStylusTipTranslationRANSAC(glm::dvec3 &translationOut, double &translationErrorOut,
			glm::dvec4 &sphere, vector<pair<glm::dvec3, glm::dquat>> &points,
			RansacConfig &config);

		bool computePlaneLSFRANSAC(glm::dvec4 &planeOut, double &fittingErrorOut,
			vector<glm::dvec3> &points,
			RansacConfig &config);
	public:
		RansacConfig defaultSphereFittingRansacConfig;	//!< used for sphere fitting
		RansacConfig defaultTipTranslationRansacConfig; //!< used for stylus tip translation calibration
		RansacConfig defaultPlaneFittingRansacConfig; //!< used for drawing plane calibration

		MagicStylus();
		~MagicStylus();

		bool calibrateStylusTipTransformation(vector<pair<glm::dvec3, glm::dquat>> &posesList);
		void resetStylusTipTransformation();
		bool getStylusTipPosition(glm::dvec3 &posOut, glm::dvec3 &basePos, glm::dquat &baseOri);

		bool calibrateDrawingPlane(vector<glm::dvec3> points);
		void resetDrawingPlane();
		glm::dvec4 getDrawingPlane();
		bool hasDrawingPlane();
		void setPlaneDistanceThreshold(double mu);
		double getPlaneDistanceThreshold();
		bool onPlane(glm::dvec3 &point);

		void saveCalibrations(string file);
		void loadCalibrations(string file);
	};
}
cmake_minimum_required(VERSION 3.16)
project(cpu_object_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# On Apple Silicon, Homebrew OpenCV lives under /opt/homebrew; on Intel under
# /usr/local. CMake usually finds it; uncomment if find_package fails:
# set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(tracker
  src/main.cpp
  src/TrackerEngine.cpp
  src/VideoSource.cpp
  src/KalmanBoxTracker.cpp
  src/TemplateMatcher.cpp
  src/MosseMatcher.cpp
  src/RotationEstimator.cpp
  src/CvTrackerMatcher.cpp
  src/SearchRegion.cpp
  src/ConfidenceEvaluator.cpp
  src/ReacquisitionManager.cpp
  src/TrackerMetrics.cpp
  src/Visualizer.cpp
)

target_include_directories(tracker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})

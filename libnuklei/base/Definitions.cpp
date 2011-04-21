// (C) Copyright Renaud Detry   2007-2011.
// Distributed under the GNU General Public License. (See
// accompanying file LICENSE.txt or copy at
// http://www.gnu.org/copyleft/gpl.html)

/** @file */


#include <string>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <nuklei/Common.h>
#include <nuklei/Math.h>

namespace nuklei {
  
  const std::string INFOSTRING = 
  std::string("Built on " __DATE__ " at " __TIME__) +
  " with `" + BUILDTYPE + "' flags.\n"
  "This application is part of the "
  "Multidimensional Features Hierarchies framework. "
  "For more information, please contact one of the following:\n"
  "Renaud.Detry@ULg.ac.be, "
  "Justus.Piater@ULg.ac.be.\n";
    
  template<typename T>
  static T initConst(const char *envVar, T def)
  {
    const char * envVal = getenv(envVar);
    if (envVal != NULL)
    {
      T val = numify<T>(envVal);
      
      const char * log = getenv("NUKLEI_LOG_LEVEL");
      if (log != NULL && numify<unsigned>(log) >= Log::INFO)
        std::cout << "export " << envVar << "=" << val
                  << " #(def: " << def << ").\n";
      return val;
    }
    else
      return def;
  }
  
#define defConst(TYPE, VAR, VAL) \
const TYPE VAR = initConst<TYPE>("NUKLEI_" #VAR, VAL)
  
  const coord_t FLOATTOL = 1e-12;
  
  defConst(coord_t, ROTATION_STDEV, .2);
  defConst(coord_t, OBSERVATION_ORIENTATION_STDEV, .4);
  defConst(coord_t, OBSERVATION_LOCATION_STDEV, 12);
  defConst(bool, OBSERVATION_COMPUTE_LOC_DIFF, false);

  // Default precision in text files.
  const int PRECISION = std::numeric_limits<double>::digits10 + 2;
  
  defConst(unsigned, N_PRIMITIVE_FEATURES, 1);
  defConst(unsigned, SPATIAL_CLUSTERING, 0);
  defConst(double, CLUSTER_SOFT_ASSIGN_THRESHOLD, .2);
  defConst(double, CLUSTER_CLUSTER_CREATE_THRESHOLD, .33);
  defConst(bool, LEARNING_ASSIGN_CLOSEST, true);

  defConst(unsigned, REFINE_SIZE, 500);
  defConst(unsigned, N_PASS, 100);
  defConst(unsigned, N_PRIMARY_PASS, 2);
  defConst(double, PROPOSAL_SIZE_FACTOR, 5);
  defConst(bool, MCMC_NBP, 0);
  defConst(unsigned, MCMC_NBP_N_CHAINS, 2);
  
  defConst(bool, KDTREE_DENSITY_EVAL, true);

  const coord_t LOCSTDEVMIN = .1;
  const coord_t ORISTDEVMIN = .04;

  defConst(weight_t, WHITE_NOISE_POWER, 1e-4);

  defConst(unsigned int, KDE_KTH_NEAREST_NEIGHBOR, 8);
  
  const bool NORMALIZE_DENSITIES = true;

  // Remember that hue/saturation are projected on a disk of diameter 2.
  // A value of 1.0 for the following constant already quite lowers its
  // importance.
  // Rangoe is [0,1].
  defConst(appear_t, HSV_METRIC_VALUE_WEIGHT, .6);

  const int NICEINC = 17;
  
  defConst(std::string, SERIALIZATION_DEFAULT_BOOST_ARCHIVE, "bxmlc");

  defConst(unsigned, IMAGE_PROJECTION_RADIUS, 3);

  defConst(bool, ENABLE_CONSOLE_BACKSPACE, true);
  
  defConst(unsigned, LOG_LEVEL, 0);
  
  // Should be moved to Log and protected by a mutex, when
  // ProgressIndicator is used in mt contexts.
  bool LAST_OUTPUT_LINE_IS_PROGRESS = false;
  
  const bool INTERACTIVE_SHELL =
    isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) && isatty(STDERR_FILENO) &&
    (getenv("TERM") != NULL) &&
    (std::string(getenv("TERM")) != "dumb");

  defConst(unsigned, N_THREADS, 0);

#ifdef NDEBUG
#warning Compiling with NDEBUG set.
#endif
}
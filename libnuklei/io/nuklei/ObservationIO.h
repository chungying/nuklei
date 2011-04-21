// (C) Copyright Renaud Detry   2007-2011.
// Distributed under the GNU General Public License. (See
// accompanying file LICENSE.txt or copy at
// http://www.gnu.org/copyleft/gpl.html)

/** @file */

#ifndef NUKLEI_OBSERVATIONSERIAL_H
#define NUKLEI_OBSERVATIONSERIAL_H

#include <typeinfo>
#include <boost/utility.hpp>

#include <nuklei/Definitions.h>
#include <nuklei/Observation.h>
#include <nuklei/RegionOfInterest.h>
#include <nuklei/nullable.h>

namespace nuklei {
  
  class KernelCollection;
  
  class ObservationIOError : public Error
    {
    public: ObservationIOError(const std::string &s) : Error(s) {}
    };


  /**
   * @author Renaud Detry <detryr@montefiore.ulg.ac.be>
   */
  class ObservationReader : boost::noncopyable
    {
    public:
      virtual ~ObservationReader();
      
      std::auto_ptr<Observation> readObservation();
      std::auto_ptr<KernelCollection> readObservations();
      void readObservations(KernelCollection &kc);
  
      virtual Observation::Type type() const = 0;

      void registerType(Observation::Type t)
      {
        oc.type_ = nameFromType<Observation>(t);
      }

      void registerType(const ObservationReader& reader)
      {
        oc.type_ = typeid(reader).name();
      }

      virtual nullable<unsigned> nObservations() const
      { return undefined(); }

      virtual void addRegionOfInterest(boost::shared_ptr<RegionOfInterest> roi);
  

      virtual void init() { registerType(*this); init_(); }
      virtual void reset() = 0;
  
      class Counter
        {
          typedef std::map<std::string, unsigned> map_t;
          typedef std::list<std::string> list_t;
        public:
          Counter() {}
        
          void incLabel(const std::string &label);
          bool empty() const;
        
          //private:
          friend std::ostream& operator<<(std::ostream &out, const Counter &c);
          std::string type_;
          map_t counts_;
          list_t labels_;
        };

      static std::auto_ptr<ObservationReader>
      createReader(const std::string& arg);
      static std::auto_ptr<ObservationReader>
      createReader(const std::string& arg, const Observation::Type t);
    protected:
      virtual std::auto_ptr<Observation> readObservation_() = 0;
      virtual void init_() = 0;
      Counter oc;
    private:
      boost::shared_ptr<RegionOfInterest> roi_;
    };

  std::ostream& operator<<(std::ostream &out, const ObservationReader::Counter &c);

  void readObservations(ObservationReader& r, KernelCollection &kc);
  
  /**
   * @author Renaud Detry <detryr@montefiore.ulg.ac.be>
   */
  class ObservationWriter : boost::noncopyable
    {
    public:
      virtual ~ObservationWriter();
      
      virtual void writeObservation(const Observation &o) = 0;
      void writeObservations(const KernelCollection &kc);

      virtual void writeBuffer() = 0;
      virtual void init() = 0;
      virtual void reset() = 0;
      virtual std::auto_ptr<Observation> templateObservation() const = 0;

      virtual Observation::Type type() const = 0;

      static std::auto_ptr<ObservationWriter>
      createWriter(const std::string& arg, const Observation::Type t);

    };

  void writeObservations(ObservationWriter &w, const KernelCollection &kc);  
}

#endif

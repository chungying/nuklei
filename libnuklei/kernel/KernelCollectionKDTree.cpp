// (C) Copyright Renaud Detry   2007-2015.
// Distributed under the GNU General Public License and under the
// BSD 3-Clause License (See accompanying file LICENSE.txt).

/** @file */

#include "KernelCollectionTypes.h"

#include <nuklei/KernelCollection.h>

#include "nanoflann.hpp"

namespace nuklei {
  
  namespace nanoflann_types
  {
    
    struct PointCloud
    {
      typedef double T;
      struct Point
      {
        Point(T x, T y, T z) : x(x), y(y), z(z) {}
        T  x,y,z;
      };
      
      std::vector<Point>  pts;
      
      // Must return the number of data points
      inline size_t kdtree_get_point_count() const { return pts.size(); }
      
      // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
      inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t size) const
      {
        const T d0=p1[0]-pts[idx_p2].x;
        const T d1=p1[1]-pts[idx_p2].y;
        const T d2=p1[2]-pts[idx_p2].z;
        return d0*d0+d1*d1+d2*d2;
      }
      
      // Returns the dim'th component of the idx'th point in the class:
      // Since this is inlined and the "dim" argument is typically an immediate value, the
      //  "if/else's" are actually solved at compile time.
      inline T kdtree_get_pt(const size_t idx, int dim) const
      {
        if (dim==0) return pts[idx].x;
        else if (dim==1) return pts[idx].y;
        else return pts[idx].z;
      }
      
      // Optional bounding-box computation: return false to default to a standard bbox computation loop.
      //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
      //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
      template <class BBOX>
      bool kdtree_get_bbox(BBOX &bb) const { return false; }
      
    };
    
    using namespace nuklei_nanoflann;
    typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud > ,
		PointCloud,
		3 /* dim */
		> KDTreeIndex;
    typedef std::pair<boost::shared_ptr<KDTreeIndex>, PointCloud> Tree;
  }
  
  void KernelCollection::buildNeighborSearchTree()
  {
    NUKLEI_TRACE_BEGIN();
#ifdef NUKLEI_USE_CGAL
    using namespace cgal_neighbor_search_types;
    
    boost::shared_ptr<Tree> tree(new Tree);
    for (const_iterator i = as_const(*this).begin(); i != as_const(*this).end(); ++i)
    {
      Vector3 loc = i->getLoc();
      tree->insert(Point_d(loc.X(), loc.Y(), loc.Z()));
    }
    if (deco_.has_key(NSTREE_KEY)) deco_.erase(NSTREE_KEY);
    deco_.insert(NSTREE_KEY, tree);
    
#else
    NUKLEI_THROW("This function requires CGAL. See http://nuklei.sourceforge.net/doxygen/group__install.html");
#endif
    NUKLEI_TRACE_END();
  }
  
  void KernelCollection::buildKdTree()
  {
    if (KDTREE_NANOFLANN)
    {
      using namespace nanoflann_types;
      
      PointCloud pc;
      for (const_iterator i = as_const(*this).begin(); i != as_const(*this).end(); ++i)
      {
        Vector3 loc = i->getLoc();
        pc.pts.push_back(PointCloud::Point(loc.X(), loc.Y(), loc.Z()));
      }
            
      boost::shared_ptr<Tree> tree(new Tree(boost::shared_ptr<KDTreeIndex>(), pc));
      tree->first = boost::shared_ptr<KDTreeIndex>(new KDTreeIndex(3 /*dim*/, tree->second, KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ));
      tree->first->buildIndex();
      
      if (deco_.has_key(KDTREE_KEY)) deco_.erase(KDTREE_KEY);
      deco_.insert(KDTREE_KEY, tree);
    }
    else
    {
      using namespace libkdtree_types;
      
      boost::shared_ptr<Tree> tree(new Tree);
      for (const_iterator i = as_const(*this).begin(); i != as_const(*this).end(); ++i)
      {
        Vector3 loc = i->getLoc();
        tree->insert(FlexiblePoint(loc.X(), loc.Y(), loc.Z(),
                                   std::distance(as_const(*this).begin(), i)));
      }
      tree->optimise();
      
      if (deco_.has_key(KDTREE_KEY)) deco_.erase(KDTREE_KEY);
      deco_.insert(KDTREE_KEY, tree);
    }
  }
  
  template<class KernelType>
  weight_t KernelCollection::staticEvaluationAt(const kernel::base &k,
                                                const EvaluationStrategy strategy) const
  {
    NUKLEI_TRACE_BEGIN();
    
    NUKLEI_ASSERT(size() > 0);
    
    NUKLEI_ASSERT(KernelType().type() == *kernelType_);
    NUKLEI_ASSERT(*kernelType_ == k.polyType());
    
    coord_t value = 0;
    if (KDTREE_DENSITY_EVAL && size() > 1000)
    {
#if NUKLEI_CHECK_KDTREE_COUNT
      int n_inside = 0;
      coord_t rng = maxLocCutPoint()*maxLocCutPoint();
      {
        const KernelType &evalPoint = static_cast<const KernelType&>(k);
        for (const_iterator i = begin(); i != end(); i++)
        {
          const KernelType &densityPoint = static_cast<const KernelType&>(*i);
          
          if ((densityPoint.loc_-evalPoint.loc_).SquaredLength() < rng)
            n_inside++;
        }
      }
#endif
      
      if (KDTREE_NANOFLANN)
      {
        using namespace nanoflann_types;
        if (!deco_.has_key(KDTREE_KEY))
          NUKLEI_THROW("Undefined kd-tree. Call buildKdTree() first.");
        
        const KernelType &evalPoint = static_cast<const KernelType&>(k);
        
        boost::shared_ptr<Tree> tree(deco_.get< boost::shared_ptr<Tree> >(KDTREE_KEY));
        //FlexiblePoint s(evalPoint.loc_.X(), evalPoint.loc_.Y(), evalPoint.loc_.Z(), -1);
        
        coord_t range = maxLocCutPoint();
        // nanoflann takes squared distances.
        range = range*range;
        
        std::vector<std::pair<size_t,coord_t> > indices_dists;
        RadiusResultSet<coord_t,size_t> resultSet(range,indices_dists);
        
        as_const(*tree).first->findNeighbors(resultSet, evalPoint.loc_, nuklei_nanoflann::SearchParams());
        
        if(indices_dists.size() == 0)
        {
          size_t num_closest = 1;
          std::pair< std::vector<size_t>,std::vector<coord_t> > indices_dists = std::make_pair(std::vector<size_t>(num_closest),std::vector<coord_t>(num_closest) );
          as_const(*tree).first->knnSearch(evalPoint.loc_, num_closest, indices_dists.first.data(), indices_dists.second.data());
          for (std::vector<size_t>::const_iterator idx = indices_dists.first.begin(); idx != indices_dists.first.end(); idx++)
          {
            coord_t cvalue = 0;
            const KernelType &densityPoint = static_cast<const KernelType&>(at(idx));
            cvalue = densityPoint.eval(evalPoint);
            if (strategy == MAX_EVAL) value = std::max(value, cvalue);
            else if (strategy == SUM_EVAL) value += cvalue;
            else if (strategy == WEIGHTED_SUM_EVAL) value += cvalue * densityPoint.getWeight();
            else NUKLEI_ASSERT(false);
          }
        }
        else
        {
          for (std::vector<std::pair<size_t,coord_t> >::const_iterator i = indices_dists.begin(); i != indices_dists.end(); i++)
          {
            coord_t cvalue = 0;
            const KernelType &densityPoint = static_cast<const KernelType&>(at(i->first));
            cvalue = densityPoint.eval(evalPoint);
            if (strategy == MAX_EVAL) value = std::max(value, cvalue);
            else if (strategy == SUM_EVAL) value += cvalue;
            else if (strategy == WEIGHTED_SUM_EVAL) value += cvalue * densityPoint.getWeight();
            else NUKLEI_ASSERT(false);
          }
        }
      }
      else
      {
        using namespace libkdtree_types;
        if (!deco_.has_key(KDTREE_KEY))
          NUKLEI_THROW("Undefined kd-tree. Call buildKdTree() first.");
        
        const KernelType &evalPoint = static_cast<const KernelType&>(k);
        std::vector<FlexiblePoint> in_range;
        
        
        boost::shared_ptr<Tree> tree(deco_.get< boost::shared_ptr<Tree> >(KDTREE_KEY));
        FlexiblePoint s(evalPoint.loc_.X(), evalPoint.loc_.Y(), evalPoint.loc_.Z(), -1);
        
        coord_t range = maxLocCutPoint();
        
        as_const(*tree).find_within_range(s, range, std::back_inserter(in_range));
        
        if(in_range.size()==0) NUKLEI_ASSERT(false);//cannot be zero neighbors
        //TODO solve the assertion

        for (std::vector<FlexiblePoint>::const_iterator i = in_range.begin(); i != in_range.end(); i++)
        {
          coord_t cvalue = 0;
          const KernelType &densityPoint = static_cast<const KernelType&>(at(i->idx()));
          cvalue = densityPoint.eval(evalPoint);
          if (strategy == MAX_EVAL) value = std::max(value, cvalue);
          else if (strategy == SUM_EVAL) value += cvalue;
          else if (strategy == WEIGHTED_SUM_EVAL) value += cvalue * densityPoint.getWeight();
          else NUKLEI_ASSERT(false);
        }
      }
    }
    else //the condition of (KDTREE_DENSITY_EVAL && size() <= 1000)
    {
      const KernelType &evalPoint = static_cast<const KernelType&>(k);
      for (const_iterator i = begin(); i != end(); i++)
      {
        coord_t cvalue = 0;
        const KernelType &densityPoint = static_cast<const KernelType&>(*i);
        //TODO if densityPooint.eval perform cutpoint, find the closest point
        cvalue = densityPoint.eval(evalPoint);
        if (strategy == MAX_EVAL) value = std::max(value, cvalue);
        else if (strategy == SUM_EVAL) value += cvalue;
        else if (strategy == WEIGHTED_SUM_EVAL) value += cvalue * densityPoint.getWeight();
        else NUKLEI_ASSERT(false);
      }
    }
    
    return value;
    NUKLEI_TRACE_END();
  }
  
  weight_t KernelCollection::evaluationAt(const kernel::base &k,
                                          const EvaluationStrategy strategy) const
  {
    NUKLEI_TRACE_BEGIN();
    if (empty()) return 0;
    NUKLEI_ASSERT(kernelType_ == k.polyType());
    
    coord_t value = 0;
    switch (*kernelType_)
    {
      case kernel::base::R3:
      {
        value = staticEvaluationAt<kernel::r3>(k, strategy);
        break;
      }
      case kernel::base::R3XS2:
      {
        value = staticEvaluationAt<kernel::r3xs2>(k, strategy);
        break;
      }
      case kernel::base::R3XS2P:
      {
        value = staticEvaluationAt<kernel::r3xs2p>(k, strategy);
        break;
      }
      case kernel::base::SE3:
      {
        value = staticEvaluationAt<kernel::se3>(k, strategy);
        break;
      }
      default:
      {
        // One could implement a fallback polymorphic eval here (polyEval), but
        // it's so slow that it's not really useful.
        NUKLEI_THROW("Unknow kernel type.");
        break;
      }
    }
    
    return value;
    NUKLEI_TRACE_END();
  }
  
}


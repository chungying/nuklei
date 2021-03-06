// Geometric Tools, LLC
// Copyright (c) 1998-2010
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.0 (2010/01/01)

#ifndef NUKLEI_WMFTABLE_H
#define NUKLEI_WMFTABLE_H

#include "Wm5Tuple.h"

// The class TYPE is either native data or is class data that has the
// following member functions:
//   TYPE::TYPE ()
//   TYPE::TYPE (const TYPE&);
//   TYPE& TYPE::operator= (const TYPE&)

namespace nuklei_wmf
{

template <int NUMROWS, int NUMCOLS, typename TYPE>
class Table
{
public:
    // Construction and destruction.  The default constructor does not
    // initialize the array elements for native elements.  The array elements
    // are initialized for class data whenever TYPE initializes during its
    // default construction.
    Table ();
    Table (const Table& table);
    ~Table ();

    // Coordinate access.
    inline operator const TYPE* () const;
    inline operator TYPE* ();
    inline const TYPE* operator[] (int row) const;
    inline TYPE* operator[] (int row);
    inline TYPE operator() (int row, int col) const;
    inline TYPE& operator() (int row, int col);
    void SetRow (int row, const Tuple<NUMROWS,TYPE>& tuple);
    Tuple<NUMROWS,TYPE> GetRow (int row) const;
    void SetColumn (int col, const Tuple<NUMCOLS,TYPE>& tuple);
    Tuple<NUMCOLS,TYPE> GetColumn (int col) const;

    // Assignment.
    Table& operator= (const Table& table);

    // Comparison.  The inequalities make the comparisons using memcmp, thus
    // treating the tuple as an array of unsigned bytes.
    bool operator== (const Table& table) const;
    bool operator!= (const Table& table) const;
    bool operator<  (const Table& table) const;
    bool operator<= (const Table& table) const;
    bool operator>  (const Table& table) const;
    bool operator>= (const Table& table) const;

protected:
    // The array is stored in row-major order.
    enum { NUMENTRIES = NUMROWS*NUMCOLS };
    TYPE mEntry[NUMENTRIES];
};

#include "Wm5Table.inl"

}

#endif

#ifndef TYPES_H
#define TYPES_H

struct Point3f
{
public:
    // Point3f Public Methods
    Point3f()
            : vec(0.f, 0.f, 0.f, 1.f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {

    }

    Point3f(float xx, float yy, float zz)
            : vec(xx, yy, zz, 1.f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {
        assert(!HasNaNs());
    }

    Point3f(const Point3f& p)
            : vec(p.vec[0], p.vec[1], p.vec[2], 1.0f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {
        assert(!p.HasNaNs());
    }


    inline const float& x() const
    {
        return vec[0];
    }

    inline float& x()
    {
        return vec[0];
    }

    inline const float& y() const
    {
        return vec[1];
    }

    inline float& y()
    {
        return vec[1];
    }

    inline const float& z() const
    {
        return vec[2];
    }

    inline float& z()
    {
        return vec[2];
    }

    bool HasNaNs() const
    {
        return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
    }

    bool operator==(const Point3f &p) const
    {

        return vec == p.vec;
    }

    bool operator!=(const Point3f &p) const
    {
        return vec != p.vec;
    }

    friend std::ostream& operator<<(std::ostream& out, const Point3f& p)
    {
        out.width(4);
        out.precision(3);
        out << p.vec[0] << ", " << p.vec[1] << ", " << p.vec[2];
        return out;
    }
    // Point3f Public Data
    Eigen::Vector4f vec;
//    float& x, &y, &z;
};

#endif


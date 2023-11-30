#ifndef DYNET_XFUNCTORS_H
#define DYNET_XFUNCTORS_H

#ifndef __CUDACC__
#include <Eigen/Eigen>
#endif

#include "dynet/functors.h"

// these functors are implemented to exploit Eigen's internal logic for doing
// vectorized arithmetic. I'm putting them in a separate file since, if Eigen
// breaks backward compatibility by changing an internal interface, I want
// the necessary changes to be localized.
//
// to implement your own functor, you need to provide
//   1) operator() implemented on the scalar data type
//   2) packetOp implemented using vector ("packet") type
//   3) the functor_traits specialization for your functor
//      that tells the compiler whether your architecture
//      has vectorized support for the operations you need
//      and an estimate of the cost of the operation

namespace dynet {
template<typename Scalar> struct const_add_op {
  const_add_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c + x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return padd(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_add_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasAdd
  };
};
} }

namespace dynet {
template<typename Scalar> struct const_minus_op {
  const_minus_op(const Scalar& c) : c(c) {}
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    return c - x;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    return psub(pset1<Packet>(c), x);
  }
  Scalar c;
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::const_minus_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub
  };
};
} }

#define EIGEN_EMPTY_STRUCT_CTOR(X)		 \
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X() {}			\
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE X(const X& ) {}

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    const Scalar one = Scalar(1.0);
    if (x >= 0.0){
        return one / (one + expf(-x));
    }else{
        return expf(x) / (one + expf(x));
    }
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1.0);
    const Packet half = pset1<Packet>(0.5);
    //return padd(pmul(half, ptanh(pmul(x, half))), half);
    return psub(padd(pmin(half, pdiv(one, padd(one, pexp(pnegate(x))))), pmax(half, pdiv(pexp(x), padd(one, pexp(x))))), half);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 3 + NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasSub && 
                   packet_traits<Scalar>::HasMax && packet_traits<Scalar>::HasNegate &&
                   packet_traits<Scalar>::HasMin && packet_traits<Scalar>::HasExp 
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_log_sigmoid_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_sigmoid_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
    using std::log1pf;
    // distinguish between positive and negative values of x for precision
    if (x>0)
        return -log1pf(expf(-x));
    else
        return x - log1pf(expf(x));
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
    using namespace Eigen::internal;
    const Packet minus_one = pset1<Packet>(-1.0);
    // Trick to mimick a condition do the computation for both cases and take the min/max with a "pivot" value (here -1) then add. Then substract the excess -1
    return pmin(
            padd(
             // Negative case (close to x)
             pmin(
                 minus_one,
                 psub(x, plog1p(pexp(x)))
                 ),
             // Positive case (close to 0)
             pmax(
                 minus_one,
                 pnegate(plog1p(pexp(pnegate(x))))
                 )
             ),
            minus_one);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_log_sigmoid_forward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost *6 + NumTraits<Scalar>::MulCost * 4,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasSub && 
                   packet_traits<Scalar>::HasMin && packet_traits<Scalar>::HasMax && 
                   packet_traits<Scalar>::HasLog1p && packet_traits<Scalar>::HasExp &&
                   packet_traits<Scalar>::HasNegate
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_log_sigmoid_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_log_sigmoid_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { 
    return (1 - expf(t)) * d;
  }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, pexp(t)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_log_sigmoid_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasExp
  };
};
}}

namespace dynet {
template<typename Scalar> struct scalar_sqrt_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_sqrt_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& t, const Scalar& d) const {
    const Scalar two = Scalar(2);
    return d / (two * t);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet two = pset1<Packet>(2);
    return pdiv(d, pmul(two, t));
  }
};
typedef scalar_sqrt_backward_op<float> FSqrtBackward;
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_sqrt_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_asinh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    using std::asinh;
#endif
    return asinh(x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    using std::asinh;
#endif
    return asinh(x);
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_acosh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    using std::acosh;
#endif
    return acosh(x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    using std::acosh;
#endif
    return acosh(x);
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_atanh_forward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_forward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x) const {
#ifndef __CUDACC__
    using std::atanh;
#endif
    return atanh(x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x) const {
#ifndef __CUDACC__
    using std::atanh;
#endif
    return atanh(x);
  }
};
}

namespace dynet {
template<typename Scalar> struct scalar_tan_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tan_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { return (1 + t * t) * d; }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(pmadd(t, t, one), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tan_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul
  };
};
}}

namespace dynet {
template<typename Scalar> struct scalar_asin_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asin_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(psub(one, pmul(x, x))), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_asin_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_acos_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acos_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return -d / sqrt(1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pnegate(pmul(prsqrt(psub(one, pmul(x, x))), d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_acos_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasNegate && packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_atan_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atan_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / (x * x + 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    //return pdiv(d, padd(pmul(x, x), one));
    return pdiv(d, pmadd(x, x, one));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_atan_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_asinh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_asinh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(x * x + 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(pmadd(x, x, one)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_asinh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasAdd && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_acosh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_acosh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / sqrt(x * x - 1);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(prsqrt(psub(pmul(x, x), one)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_acosh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 10,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasRsqrt
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_atanh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_atanh_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    return d / (1 - x * x);
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pdiv(d, psub(one, pmul(x, x)));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_atanh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost * 2 + NumTraits<Scalar>::MulCost * 3,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul &&
                   packet_traits<Scalar>::HasDiv
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_erf_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erf_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& d) const {
    const Scalar sqrt_pi_over2(1.1283791670955125738961589);
    return sqrt_pi_over2 * expf(-x * x) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet sqrt_pi_over2 = pset1<Packet>(1.1283791670955125738961589);
    return pmul(sqrt_pi_over2, pmul(pexp(pnegate(pmul(x, x))), d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_erf_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::MulCost * 8,
    PacketAccess = packet_traits<Scalar>::HasExp && packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasNegate
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_logistic_sigmoid_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_logistic_sigmoid_backward_op)
  DYNET_DEVICE_FUNC inline const Scalar operator() (const Scalar& t, const Scalar& d) const {
    const Scalar one = Scalar(1);
    return (one - t) * t * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC inline Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, t), pmul(t, d));
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_logistic_sigmoid_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + NumTraits<Scalar>::MulCost * 2,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
} }

namespace dynet {
template<typename Scalar> struct scalar_tanh_backward_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_tanh_backward_op)
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& t, const Scalar& d) const { return (1 - t * t) * d; }
  template<typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t, const Packet& d) const {
    using namespace Eigen::internal;
    const Packet one = pset1<Packet>(1);
    return pmul(psub(one, pmul(t, t)), d);
  }
};
}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_tanh_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasMul
  };
};
}}

namespace dynet {
//this is slower than the dumb implementation, probably because of the pset operations
// which could be factored out into the constructor, but the Packet type isn't used
// then (and I think fixing this would be hard)
template<typename Scalar> struct scalar_nlsoftmax_backward_op {
  scalar_nlsoftmax_backward_op(const Scalar& lz, const Scalar& err) : logz(lz), d(err) {}
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator()(const Scalar& t) const {
    return expf(t - logz) * d;
  }
  template <typename Packet>
  DYNET_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& t) const {
    using namespace Eigen::internal;
    const Packet lz = pset1<Packet>(logz);
    const Packet dd = pset1<Packet>(d);
    return pmul(pexp(psub(t, lz)), dd);
  }
  Scalar logz;
  Scalar d;
};}

namespace Eigen { namespace internal {
template<typename Scalar>
struct functor_traits<dynet::scalar_nlsoftmax_backward_op<Scalar> > {
  enum {
    Cost = NumTraits<Scalar>::AddCost + 6 * NumTraits<Scalar>::MulCost,
    PacketAccess = packet_traits<Scalar>::HasSub && packet_traits<Scalar>::HasExp
  };
};
}}

#endif

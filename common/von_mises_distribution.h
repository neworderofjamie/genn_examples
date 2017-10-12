#pragma once

// Standard C includes
#include <cmath>

//----------------------------------------------------------------------------
// VonMisesDistribution
//----------------------------------------------------------------------------
//!< Von Mises distribution object with a mean of mu and a concentration of kappa
//!< Uses the acceptance-rejection sampling algorithm proposed by Best and Fisher (1979)
template<typename T>
class VonMisesDistribution
{
public:
    VonMisesDistribution(T mu, T kappa) : m_Mu(mu), m_Kappa(kappa)
    {
        const T a = 1.0 + sqrt(1.0 + (4.0 * kappa * kappa));
        const T b = (a - sqrt(2.0 * a)) / (2.0 * kappa);

        // **NOTE** ideally m_R would also be const but using delegating
        // constructors or whatever would be super-convoluted
        m_R = (1.0 + (b * b)) / (2.0 * b);
    }

    //----------------------------------------------------------------------------
    // Operators
    //----------------------------------------------------------------------------
    template<typename Generator>
    T operator()(Generator &generator) const
    {
        constexpr double pi = 3.141592653589793238462643383279502884;
        std::uniform_real_distribution<T> distribution(0.0, 1.0);

        double u1, u2, u3;
        double z, f, c;
        while(true)
        {
            // Generate 3 uniformly-distributed random numbers
            u1 = distribution(generator);
            u2 = distribution(generator);
            u3 = distribution(generator);

            z = cos(pi * u1);
            f = (1.0 + (m_R * z)) / (m_R + z);
            c = m_Kappa * (m_R - f);

            if(((c * (2.0 - c)) - u2) > 0.0) {
                break;
            }

            if((log(c / u2) + 1.0 - c) >= 0.0) {
                break;
            }
        }

        return m_Mu + sign(u3 - 0.5) * acos(f);
    }

private:
    //----------------------------------------------------------------------------
    // Static methods
    //----------------------------------------------------------------------------
    static T sign(T x)
    {
        if (x > 0) {
            return 1;
        }
        else if (x < 0) {
            return -1;
        }
        else {
            return 0;
        }
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    const T m_Mu;
    const T m_Kappa;
    T m_R;
};
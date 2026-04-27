/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2004-2011 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
                            | Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::RASModels::FG_MoE

Group
    grpRASTurbulence

Description
    Standard high Reynolds-number k-omega turbulence model for
    incompressible and compressible flows.

    References:
    \verbatim
        Wilcox, D. C. (1998).
        Turbulence modeling for CFD
        (Vol. 2, pp. 103-217). La Canada, CA: DCW industries.
    \endverbatim

    The default model coefficients are
    \verbatim
        FG_MoECoeffs
        {
            Cmu         0.09;  // Equivalent to betaStar
            alpha       0.52;
            beta        0.072;
            alphak      0.5;
            alphaOmega  0.5;
        }
    \endverbatim

SourceFiles
    FG_MoE.C

\*---------------------------------------------------------------------------*/

#ifndef FG_MoE_H
#define FG_MoE_H

#include "RASModel.H"
#include "nonlinearEddyViscosity.H"
#include "simpleFilter.H"
#include "LESdelta.H"

/*The following stuff is for Python interoperability*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL POD_ARRAY_API
#include <numpy/arrayobject.h>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                           Class FG_MoE Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class FG_MoE
:
    //public eddyViscosity<RASModel<BasicTurbulenceModel>>
    public nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
{

protected:

    // Protected data
            dictionary curDict_;

            //- whether to use the baseline EARSM05 model
            Switch useBaselineModel;

            //- whether the case is 2D and being empty in Y or Z directions
            Switch homogeneousZ;
            Switch homogeneousY;

            //- freestream values of k and omega for decay control
            //- set as zero for internal cases (e.g. channel flow)
            dimensionedScalar kInf_;
            dimensionedScalar omegaInf_;

            //- standard model coefficients
            dimensionedScalar alphaK1_;
            dimensionedScalar alphaK2_;
            dimensionedScalar alphaOmega1_;
            dimensionedScalar alphaOmega2_;
            dimensionedScalar gamma1_;
            dimensionedScalar gamma2_;
            dimensionedScalar beta1_;
            dimensionedScalar beta2_;
            dimensionedScalar sigmaD1_;
            dimensionedScalar sigmaD2_;
            dimensionedScalar betaStar_;
            dimensionedScalar c1_Plim;

            //- Flag to include the F3 term
            Switch F3_;


        // Fields

            volScalarField k_;
            volScalarField omega_;
            volScalarField C_mu_eff;
            volScalarField Gamma_w;
            volScalarField tau_;
            volScalarField dwall;
            volSymmTensorField S;
            volSymmTensorField aij_;
            volSymmTensorField nonlinearR;

            // ML
            volScalarField g1_;
            volScalarField g2_;
            volScalarField g3_;
            volScalarField c1_;
            volScalarField c2_;
            volScalarField theta1_Scaled_;
            volScalarField theta2_Scaled_;
            volScalarField theta3_Scaled_;
            volScalarField theta4_Scaled_;
            volScalarField theta5_Scaled_;
            volScalarField Q_sep;
            volScalarField Q_3D;
            volScalarField Q_mix;
            volScalarField p_e;
            PyObject *pName, *pModule;
            PyObject *ml_func, *ml_func_args;
            PyObject *array_2d;
            PyArrayObject *pValue;


            const volScalarField& y_ ;





    // Protected Member Functions

        virtual tmp<volScalarField> F1(const volScalarField& CDkOmega) const;
        virtual tmp<volScalarField> calcGamma(const volScalarField& CDkOmega) const;
        virtual tmp<volScalarField> F2() const;
        virtual tmp<volScalarField> F3() const;
        virtual tmp<volScalarField> F23() const;

        //- Return the blended field
        tmp<volScalarField> blend
        (
            const volScalarField& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        //- Return the internal blended field
        tmp<volScalarField::Internal> blend
        (
            const volScalarField::Internal& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        tmp<volScalarField> alphaK(const volScalarField& F1) const
        {
            return blend(F1, alphaK1_, alphaK2_);
        }

        tmp<volScalarField> alphaOmega(const volScalarField& F1) const
        {
            return blend(F1, alphaOmega1_, alphaOmega2_);
        }

        tmp<volScalarField::Internal> beta
        (
            const volScalarField::Internal& F1
        ) const
        {
            return tmp<volScalarField::Internal>::New
            (
                this->type() + ":beta",
                blend(F1, beta1_, beta2_)
            );
        }

        tmp<volScalarField::Internal> gamma
        (
            const volScalarField::Internal& F1
        ) const
        {
            return tmp<volScalarField::Internal>::New
            (
                this->type() + ":gamma",
                blend(F1, gamma1_, gamma2_)
            );
        }

        tmp<volScalarField::Internal> sigmaD
        (
            const volScalarField::Internal& F1
        ) const
        {
            return tmp<volScalarField::Internal>::New
            (
                this->type() + ":sigmaD",
                blend(F1, sigmaD1_, sigmaD2_)
            );
        }



        virtual void correctNut();
        virtual void correctNonlinearStress(const volTensorField& gradU);
        


        //- Return square of strain rate
        virtual tmp<volScalarField> S2
        (
            const volTensorField& gradU
        ) const;

        //- Return k production rate
        virtual tmp<volScalarField::Internal> Pk
        (
            const volScalarField::Internal& G
        ) const;

        //- Return epsilon/k which for standard RAS is betaStar*omega
        virtual tmp<volScalarField::Internal> epsilonByk
        (
            const volScalarField& F1,
            const volTensorField& gradU
        ) const;

        //- Return (G/nu)_0
        virtual tmp<volScalarField::Internal> GbyNu0
        (
            const volTensorField& gradU,
            const volScalarField& S2
        ) const;

        //- Return G/nu
        virtual tmp<volScalarField::Internal> GbyNu
        (
            const volScalarField::Internal& GbyNu0,
            const volScalarField::Internal& F2,
            const volScalarField::Internal& S2
        ) const;

        virtual tmp<fvScalarMatrix> kSource() const;

        virtual tmp<fvScalarMatrix> omegaSource() const;

        virtual tmp<fvScalarMatrix> Qsas
        (
            const volScalarField::Internal& S2,
            const volScalarField::Internal& gamma,
            const volScalarField::Internal& beta
        ) const;



public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("FG_MoE");


    // Constructors

        //- Construct from components
        FG_MoE
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~FG_MoE()
    {}


    // Member Functions

        //- Read RASProperties dictionary
        virtual bool read();


        //- Return the effective diffusivity for k
        tmp<volScalarField> DkEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>::New
            (
                "DkEff",
                alphaK(F1)*this->nut_ + this->nu()
            );
        }

        //- Return the effective diffusivity for omega
        tmp<volScalarField> DomegaEff(const volScalarField& F1) const
        {
            return tmp<volScalarField>::New
            (
                "DomegaEff",
                alphaOmega(F1)*this->nut_ + this->nu()
            );
        }

        //- Return the effective diffusivity for k, use k/omega as nut
        tmp<volScalarField> DkEff0(const volScalarField& F1, const volScalarField& nut0) const
        {
            return tmp<volScalarField>::New
            (
                "DkEff",
                alphaK(F1)*nut0 + this->nu()
            );
        }

        //- Return the effective diffusivity for omega, use k/omega as nut
        tmp<volScalarField> DomegaEff0(const volScalarField& F1, const volScalarField& nut0) const
        {
            return tmp<volScalarField>::New
            (
                "DomegaEff",
                alphaOmega(F1)*nut0 + this->nu()
            );
        }

        //- Return the turbulence kinetic energy
        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        //- Return the turbulence specific dissipation rate
        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        
        

        //- Solve the turbulence equations and correct the turbulence viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
#ifdef NoRepository
    #include "FG_MoE.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

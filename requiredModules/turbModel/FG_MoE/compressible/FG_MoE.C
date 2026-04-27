/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2004-2010, 2019 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
                            | Copyright (C) 2011-2017 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "FG_MoE.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"
#include "fvCFD.H"
#include "fvcSmooth.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> FG_MoE<BasicTurbulenceModel>::F1
(
    const volScalarField& CDkOmega
) const
{

    tmp<volScalarField> Gamma1 = (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_);
    tmp<volScalarField> Gamma2 = scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_);
    tmp<volScalarField> Gamma3 = 2.0*k_/max(sqr(y_)*CDkOmega, 0.0*kInf_ );

    tmp<volScalarField> Gamma = min(max(Gamma1, Gamma2), Gamma3);

    return tanh(1.5*pow4(Gamma));
}

template<class BasicTurbulenceModel>
tmp<volScalarField> FG_MoE<BasicTurbulenceModel>::calcGamma
(
    const volScalarField& CDkOmega
) const
{
    tmp<volScalarField> Gamma1 = sqrt(k_)/(betaStar_*omega_*y_);
    tmp<volScalarField> Gamma2 = scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_);
    tmp<volScalarField> Gamma3 = 2.0*k_/max(sqr(y_)*CDkOmega, 0.0*kInf_ );

    tmp<volScalarField> Gamma = min(max(Gamma1, Gamma2), Gamma3);

    return Gamma;
}


template<class BasicTurbulenceModel>
tmp<volScalarField> FG_MoE<BasicTurbulenceModel>::F2() const
{
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> FG_MoE<BasicTurbulenceModel>::F3() const
{
    tmp<volScalarField> arg3 = min
    (
        150*(this->mu()/this->rho_)/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> FG_MoE<BasicTurbulenceModel>::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


template<class BasicTurbulenceModel>
void FG_MoE<BasicTurbulenceModel>::correctNut()
{
    correctNonlinearStress(fvc::grad(this->U_));
}


template<class BasicTurbulenceModel>
void FG_MoE<BasicTurbulenceModel>::correctNonlinearStress(const volTensorField& gradU)
{
    tau_ = 1.0/betaStar_/(omega_);
            
    volTensorField W = -tau_ * skew(gradU);
    volScalarField taudivU = fvc::div( this->U_ ) * tau_;

    dwall = y_;

    if (homogeneousZ)
    {
        S = tau_ * symm(gradU);
        volScalarField trS = S.component(0) + S.component(3);
        S.replace(symmTensor::XX, S.component(symmTensor::XX) - 0.5*trS);
        S.replace(symmTensor::YY, S.component(symmTensor::YY) - 0.5*trS);
    }
    else if (homogeneousY)
    {
        S = tau_ * symm(gradU);
        volScalarField trS = S.component(0) + S.component(5);
        S.replace(symmTensor::XX, S.component(symmTensor::XX) - 0.5*trS);
        S.replace(symmTensor::ZZ, S.component(symmTensor::ZZ) - 0.5*trS);
    }
    else
    {
        S = tau_ * dev(symm(gradU));
    }

    //system rotation correction
    //W += 13.0/4.0*tau_*dimensionedTensor(dimless/dimTime, tensor(0,-Omega3,Omega2,Omega3,0,-Omega1,-Omega2,Omega1,0));

    volScalarField IIs  = tr(S & S);
    volScalarField IIw = tr(W & W);
    volScalarField IIIs = tr(S & S & S);
    volScalarField IV = tr(S & W & W);
    volScalarField V = tr(S & S & W & W);

    theta1_Scaled_ = IIs / (1.0 + mag(IIs));
    theta2_Scaled_ = IIw / (1.0 + mag(IIw));
    theta3_Scaled_ = IIIs / (1.0 + mag(IIIs));
    theta4_Scaled_ = IV / (1.0 + mag(IV));
    theta5_Scaled_ = V / (1.0 + mag(V));


    Q_sep = mag(IIs+IIw);
    Q_3D = mag(IIIs) + mag(IV);
    Q_mix = Gamma_w;


    if (!useBaselineModel)
    {
        // - construct theta as a 2D array
        int num_cells = this->mesh_.cells().size();
        double input_vals[num_cells][8];
        forAll(k_.internalField(), celli)
        {
            input_vals[celli][0] = theta1_Scaled_[celli];
            input_vals[celli][1] = theta2_Scaled_[celli];
            input_vals[celli][2] = theta3_Scaled_[celli];
            input_vals[celli][3] = theta4_Scaled_[celli];
            input_vals[celli][4] = theta5_Scaled_[celli];
            input_vals[celli][5] = Q_mix[celli];
            input_vals[celli][6] = Q_sep[celli];
            input_vals[celli][7] = Q_3D[celli];
        }
        npy_intp dim[] = {num_cells, 8};    //rows and columns of array_2d
        array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);
        PyTuple_SetItem(ml_func_args, 0, array_2d); 
        pValue = (PyArrayObject *)PyObject_CallObject(ml_func, ml_func_args);
        forAll(k_.internalField(), celli)
        {
            double *temp1 = (double *)PyArray_GETPTR2(pValue, celli, 0);
            double *temp2 = (double *)PyArray_GETPTR2(pValue, celli, 1);
            double *temp3 = (double *)PyArray_GETPTR2(pValue, celli, 2);
            g1_[celli] = (*temp1);
            g2_[celli] = (*temp2);
            g3_[celli] = (*temp3);
        }
        // release memory
        Py_DECREF(pValue);
    
    
        // dealing with the bundary
        forAll(k_.boundaryField(), patchi) 
        {
            label nFaces = this->mesh_.boundaryMesh()[patchi].size() ;
            double input_vals[nFaces][8];
            forAll(k_.boundaryField()[patchi], facei)
            {
                input_vals[facei][0] = theta1_Scaled_.boundaryField()[patchi][facei];
                input_vals[facei][1] = theta2_Scaled_.boundaryField()[patchi][facei];
                input_vals[facei][2] = theta3_Scaled_.boundaryField()[patchi][facei];
                input_vals[facei][3] = theta4_Scaled_.boundaryField()[patchi][facei];
                input_vals[facei][4] = theta5_Scaled_.boundaryField()[patchi][facei];
                input_vals[facei][5] = Q_mix.boundaryField()[patchi][facei];
                input_vals[facei][6] = Q_sep.boundaryField()[patchi][facei];
                input_vals[facei][7] = Q_3D.boundaryField()[patchi][facei];
            }
            npy_intp dim[] = {nFaces, 8};    //rows and columns of array_2d
            array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);
            // - calculate g from ML
            PyTuple_SetItem(ml_func_args, 0, array_2d); 
            pValue = (PyArrayObject *)PyObject_CallObject(ml_func, ml_func_args);
            forAll(k_.boundaryField()[patchi], facei)
            {
                double *temp1 = (double *)PyArray_GETPTR2(pValue, facei, 0);
                double *temp2 = (double *)PyArray_GETPTR2(pValue, facei, 1);
                double *temp3 = (double *)PyArray_GETPTR2(pValue, facei, 2);
                g1_.boundaryFieldRef()[patchi][facei] = (*temp1);
                g2_.boundaryFieldRef()[patchi][facei] = (*temp2);
                g3_.boundaryFieldRef()[patchi][facei] = (*temp3);
            }
            Py_DECREF(pValue);
        }
    }
    
    c1_ = g1_;
    c2_ = g2_;
    
    volScalarField A1 = 88.0/15.0/(7.0*c2_+1.0);
    volScalarField A2 = (5.0-9.0*c2_)/(7.0*c2_+1.0);
    volScalarField A3 = 11.0/(7.0*c2_+1.0)*(c1_ - 1.0);
    volScalarField A4 = 11.0/(7.0*c2_+1.0);

    volScalarField N_eq = A3 + A4;
    volScalarField beta_1eq = -A1*N_eq/(sqr(N_eq) - 2.0*IIw - 2.0/3.0*sqr(A2)*IIs);
    volScalarField C_D = 4.0/15.0/0.09+1.0-c1_;
    A3 += 11.0/(7.0*c2_+1.0) * C_D * max(1.0 + beta_1eq * IIs, 0.0);

    volScalarField P1 = (sqr(A3)/27.0 + (A1*A4/6.0 - 2.0/9.0*sqr(A2))*IIs - (2.0/3.0)*IIw)*A3;
    volScalarField P2 = sqr(P1) - pow((sqr(A3)/9.0 + (A1*A4/3.0 + 2.0/9.0*sqr(A2))*IIs + (2.0/3.0)*IIw), 3.0);

    volScalarField N_positive = A3/3.0 + cbrt(P1 + sqrt(max(P2, SMALL))) + sign(P1-sqrt(max(P2, SMALL)))*cbrt(mag(P1 - sqrt(max(P2, SMALL))));
    volScalarField temp_P1P2 = max(dimensionedScalar(dimless, SMALL), sqr(P1) - P2);
    volScalarField temp_acos = max(dimensionedScalar(dimless, -1.0), min(dimensionedScalar(dimless, 1.0), P1/sqrt(temp_P1P2)));
    volScalarField N_negative = A3/3.0 + 2.0*pow(temp_P1P2, 1.0/6.0) * cos((1.0/3.0) *acos(temp_acos));
    volScalarField N = (sign(P2)+1.0)/2.0*N_positive - (sign(P2)-1.0)/2.0*N_negative;

    p_e = (N - A3)/A4;


    volScalarField Q_ = 3.0*pow(N,5.0)
                        + (-7.5*IIw-3.5*sqr(A2)*IIs)*pow(N,3.0)
                        + (21.0*A2*IV-pow(A2,3.0)*IIIs)*sqr(N)
                        + (3.0*sqr(IIw)-8.0*IIs*IIw*sqr(A2)+24.0*sqr(A2)*V+pow(A2,4.0)*sqr(IIs))*N
                        + 2.0/3.0*pow(A2,5.0)*IIs*IIIs + 2.0*pow(A2,3.0)*IIs*IV - 2.0*pow(A2,3.0)*IIw*IIIs - 6.0*IV*A2*IIw;

    volScalarField beta_1 = -0.5*A1*N*(30.0*A2*IV-21.0*N*IIw-2.0*pow(A2,3.0)*IIIs+6.0*pow(N,3.0)-3.0*sqr(A2)*IIs*N) / Q_;
    volScalarField beta_2 = -A1*A2*(6.0*A2*IV+12.0*N*IIw+2.0*pow(A2,3.0)*IIIs-6.0*pow(N,3.0)+3.0*sqr(A2)*IIs*N) / Q_;
    volScalarField beta_3 = -3.0*A1*(2.0*sqr(A2)*IIIs + 3.0*N*A2*IIs + 6.0*IV)/Q_;
    volScalarField beta_4 = -A1*(2.0*pow(A2,3.0)*IIIs + 3.0*sqr(A2)*N*IIs + 6.0*A2*IV - 6.0*N*IIw + 3.0*pow(N,3.0))/Q_;
    volScalarField beta_5 = 9.0*A1*A2*sqr(N)/Q_;
    volScalarField beta_6 = -9.0*A1*sqr(N)/Q_;
    volScalarField beta_7 = 18.0*A1*A2*N/Q_;
    volScalarField beta_8 = 9.0*A1*sqr(A2)*N/Q_;
    volScalarField beta_9 = 9.0*A1*N/Q_;


    volTensorField WW = W & W;
    volTensorField SS = S & S;
    volTensorField SW = S & W;
    volTensorField WS = W & S;
    volTensorField SSW = SS & W;
    volTensorField WSS = W & SS;
    volTensorField SWW = SW & W;
    volTensorField WWS = WW & S;
    volTensorField SSWW = SS & WW;
    volTensorField WWSS = WW & SS;
    volTensorField SWSS = SW & SS;
    volTensorField SSWS = SS & WS;
    volTensorField WSWW = (WS & W) & W;
    volTensorField WWSW = (WW & S) & W;
    aij_ = symm(  beta_2 * (SS - (1.0/3.0)*IIs*I) 
                    + beta_3 * (WW - (1.0/3.0)*IIw*I) 
                    + beta_4 * (SW - WS) 
                    + beta_5 * (SSW - WSS) 
                    + beta_6 * (SWW + WWS - (2.0/3.0)*IV*I - IIw*S ) 
                    + beta_7 * (SSWW + WWSS - (2.0/3.0)*V*I - IV*S)
                    + beta_8 * (SWSS - SSWS)
                    + beta_9 * (WSWW - WWSW)
                    );
    
    C_mu_eff = -0.5*(beta_1 + beta_6*IIw + beta_7*IV);
    
    this->nut_ = C_mu_eff / betaStar_ * k_/omega_ ;

    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);
    BasicTurbulenceModel::correctNut();
    nonlinearR = aij_ * k_;
    this->nonlinearStress_ = nonlinearR;
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volScalarField> FG_MoE<BasicTurbulenceModel>::S2
(
    const volTensorField& gradU
) const
{
    return 2*magSqr(symm(gradU));
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> FG_MoE<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_Plim*betaStar_)*this->k_()*this->omega_());
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> FG_MoE<BasicTurbulenceModel>::epsilonByk
(
    const volScalarField& /* F1 not used */,
    const volTensorField& /* gradU not used */
) const
{
    return betaStar_*omega_();
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> FG_MoE<BasicTurbulenceModel>::GbyNu0
(
    const volTensorField& gradU,
    const volScalarField& /* S2 not used */
) const
{
    return tmp<volScalarField::Internal>::New
    (
        IOobject::scopedName(this->type(), "GbyNu"),
        gradU() && dev(twoSymm(gradU()))
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> FG_MoE<BasicTurbulenceModel>::GbyNu
(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2
) const
{
    //return GbyNu0;
    return min
    (
        GbyNu0,
        c1_Plim*betaStar_*omega_()*omega_()
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> FG_MoE<BasicTurbulenceModel>::kSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        k_,
        dimVolume*this->rho_.dimensions()*k_.dimensions()/dimTime
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> FG_MoE<BasicTurbulenceModel>::omegaSource() const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> FG_MoE<BasicTurbulenceModel>::Qsas
(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta
) const
{
    return tmp<fvScalarMatrix>::New
    (
        omega_,
        dimVolume*this->rho_.dimensions()*omega_.dimensions()/dimTime
    );
}




// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
FG_MoE<BasicTurbulenceModel>::FG_MoE
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    curDict_(this->subOrEmptyDict("RAS")),
    useBaselineModel(curDict_.lookupOrDefault<Switch>("useBaselineModel",false)),
    homogeneousZ(curDict_.lookupOrDefault<Switch>("homogeneousZ",false)),
    homogeneousY(curDict_.lookupOrDefault<Switch>("homogeneousY",false)),
    kInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "kInf",
            this->coeffDict_,
            dimVelocity*dimVelocity,
            0
        )
    ),
    omegaInf_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "omegaInf",
            this->coeffDict_,
            dimless/dimTime,
            0
        )
    ),
    alphaK1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            1.1
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.1
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.53
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            1.0
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            0.518
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.0747
        )
    ),
    beta2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    sigmaD1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmaD1",
            this->coeffDict_,
            1.0
        )
    ),
    sigmaD2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmaD2",
            this->coeffDict_,
            0.4
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    c1_Plim
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "c1_Plim",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::getOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
        )
    ),
    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    C_mu_eff
    (
        IOobject
        (
            IOobject::groupName("C_mu_eff", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("C_mu_eff", dimless, 0.09)
    ),
    Gamma_w
    (
        IOobject
        (
            IOobject::groupName("Gamma_w", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("Gamma_w", dimless, 1.0)
    ),
    tau_
    (
        IOobject
        (
            IOobject::groupName("tau_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("tau_", dimTime, 1.0)
    ),
    dwall
    (
        IOobject
        (
            IOobject::groupName("dwall", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("dwall", dimLength, 1.0)
    ),
    S
    (
        IOobject
        (
            IOobject::groupName("S", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor(dimless, Zero)
    ),
    aij_
    (
        IOobject
        (
            IOobject::groupName("aij", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor(dimless, Zero)
    ),
    nonlinearR
    (
        IOobject
        (
            IOobject::groupName("nonlinearR", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedSymmTensor(dimVelocity*dimVelocity, Zero)
    ),
    g1_
    (
        IOobject
        (
            IOobject::groupName("g1_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("g1_", dimless, 1.8)
    ),
    g2_
    (
        IOobject
        (
            IOobject::groupName("g2_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("g2_", dimless, 0.5555555555555556)
    ),
    g3_
    (
        IOobject
        (
            IOobject::groupName("g3_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("g3_", dimless, 0.0)
    ),
    c1_
    (
        IOobject
        (
            IOobject::groupName("c1_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("c1_", dimless, 1.8)
    ),
    c2_
    (
        IOobject
        (
            IOobject::groupName("c2_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("c2_", dimless, 0.5555555555555556)
    ),
    theta1_Scaled_
    (
        IOobject
        (
            IOobject::groupName("theta1_Scaled_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("theta1_Scaled_", dimless, 0.0)
    ),
    theta2_Scaled_
    (
        IOobject
        (
            IOobject::groupName("theta2_Scaled_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("theta2_Scaled_", dimless, 0.0)
    ),
    theta3_Scaled_
    (
        IOobject
        (
            IOobject::groupName("theta3_Scaled_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("theta3_Scaled_", dimless, 0.0)
    ),
    theta4_Scaled_
    (
        IOobject
        (
            IOobject::groupName("theta4_Scaled_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("theta4_Scaled_", dimless, 0.0)
    ),
    theta5_Scaled_
    (
        IOobject
        (
            IOobject::groupName("theta5_Scaled_", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("theta5_Scaled_", dimless, 0.0)
    ),
    Q_sep
    (
        IOobject
        (
            IOobject::groupName("Q_sep", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("Q_sep", dimless, 0.0)
    ),
    Q_3D
    (
        IOobject
        (
            IOobject::groupName("Q_3D", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("Q_3D", dimless, 0.0)
    ),
    Q_mix
    (
        IOobject
        (
            IOobject::groupName("Q_mix", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("Q_mix", dimless, 0.0)
    ),
    p_e
    (
        IOobject
        (
            IOobject::groupName("p_e", U.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("p_e", dimless, 0.0)
    ),
    y_(wallDist::New(this->mesh_).y())
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);


    if (type == typeName)
    {
        this->printCoeffs(type);
    }



    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    import_array1();
    pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    pModule = PyImport_Import(pName);
    ml_func = PyObject_GetAttrString(pModule, "ml_func");
    ml_func_args = PyTuple_New(1);

}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //


template<class BasicTurbulenceModel>
bool FG_MoE<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        alphaK1_.readIfPresent(this->coeffDict());
        alphaK2_.readIfPresent(this->coeffDict());
        alphaOmega1_.readIfPresent(this->coeffDict());
        alphaOmega2_.readIfPresent(this->coeffDict());
        gamma1_.readIfPresent(this->coeffDict());
        gamma2_.readIfPresent(this->coeffDict());
        beta1_.readIfPresent(this->coeffDict());
        beta2_.readIfPresent(this->coeffDict());
        sigmaD1_.readIfPresent(this->coeffDict());
        sigmaD2_.readIfPresent(this->coeffDict());
        betaStar_.readIfPresent(this->coeffDict());
        c1_Plim.readIfPresent(this->coeffDict());
        F3_.readIfPresent("F3", this->coeffDict());
        kInf_.readIfPresent("kInf", this->coeffDict());
        omegaInf_.readIfPresent("omegaInf", this->coeffDict());


        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void FG_MoE<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    volScalarField& nut = this->nut_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))
    );

    

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volScalarField S2(this->S2(tgradU()));
    volScalarField::Internal GbyNu0(this->GbyNu0(tgradU(), S2));
    const volScalarField::Internal nonlinearG
    (
        this->nonlinearStress_ && tgradU()
    );
    volScalarField::Internal G(this->GName(), nut*GbyNu0 - nonlinearG);

    // Update omega and G at the wall
    omega_.boundaryFieldRef().updateCoeffs();

    const volScalarField CDkOmega
    (
        max((fvc::grad(k_) & fvc::grad(omega_))/omega_,
            dimensionedScalar(dimless/sqr(dimTime), 0.0))
    );

    const volScalarField F1(this->F1(CDkOmega));
    Gamma_w = this->calcGamma(CDkOmega);
    const volScalarField F23(this->F23());
    const volScalarField::Internal CmueffByCmu(C_mu_eff/betaStar_);

    {
        const volScalarField::Internal gamma(this->gamma(F1));
        const volScalarField::Internal sigmaD(this->sigmaD(F1));
        const volScalarField::Internal beta((scalar(1.0) + g3_)*(F1*(beta1_ - beta2_) + beta2_));

        GbyNu0 = GbyNu(GbyNu0, F23(), S2());

        // Turbulent frequency equation
        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
            alpha()*rho()*gamma* (CmueffByCmu * GbyNu0 - nonlinearG / k_ * omega_)
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*gamma*divU, omega_)
          - fvm::Sp(alpha()*rho()*beta*omega_(), omega_)
          + fvm::SuSp
            (
                alpha()*rho()*sigmaD*CDkOmega()/omega_(),
                omega_
            )
          + alpha()*rho()*beta*sqr(omegaInf_)
          + Qsas(S2(), gamma, beta)
          + omegaSource()
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }

    {
        // Turbulent kinetic energy equation
        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(alpha, rho, k_)
          + fvm::div(alphaRhoPhi, k_)
          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
         ==
          //  alpha()*rho()*Pk(G)
            alpha()*rho()*G
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
          - fvm::Sp(alpha()*rho()*epsilonByk(F1, tgradU()), k_)
          + alpha()*rho()*betaStar_*omegaInf_*kInf_
          + kSource()
          + fvOptions(alpha, rho, k_)
        );


        kEqn.ref().relax();
        fvOptions.constrain(kEqn.ref());
        solve(kEqn);
        fvOptions.correct(k_);
        bound(k_, this->kMin_);
    }


    correctNonlinearStress(tgradU);

    tgradU.clear();


}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //

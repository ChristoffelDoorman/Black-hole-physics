#include "mpi.h"
#include <cstdio>
#include "kadath_spheric.hpp"

using namespace Kadath ;
int main (int argc, char** argv) {

    if (argc <2) {
        cout <<"File missing..." << endl ;
        abort() ;
    }

    int r2_res, r3_res ;
    double r0, r1, r2, r3, t, ome ;

    char* name_fich = argv[1] ;
    FILE* fich = fopen(name_fich, "r") ;

    Space_spheric space (fich) ;
    fread_be (&r2_res, sizeof(int), 1, fich) ;
    fread_be (&r3_res, sizeof(int), 1, fich) ;
    fread_be (&r0, sizeof(double), 1, fich) ;
    fread_be (&r1, sizeof(double), 1, fich) ;
    fread_be (&r2, sizeof(double), 1, fich) ;
    fread_be (&r3, sizeof(double), 1, fich) ;
    fread_be (&t, sizeof(double), 1, fich) ;
    fread_be (&ome, sizeof(double), 1, fich) ;
    Scalar conf (space, fich) ;
    Scalar lapse (space, fich) ;
    Vector shift (space, fich) ;
    fclose(fich) ;

    cout << "reading success" << endl ;

    // get number of domains
    int ndom = space.get_nbr_domains() ;

    // Computation adm mass :
    Val_domain integ_adm (conf(ndom-1).der_r()) ;
    double adm = -space.get_domain(ndom-1)->integ(integ_adm, OUTER_BC)/2/M_PI ;

    cout << "adm mass: " << adm << endl ;

    // Computation of Komar :
    Val_domain integ_komar (lapse(ndom-1).der_r()) ;
    double komar = space.get_domain(ndom-1)->integ(integ_komar, OUTER_BC)/4/M_PI ;

    cout << "Komar mass: " << komar << endl ;


    char name[100] ;
    sprintf (name, "schwSyst_%d_%d_%.1f_%.0f_%.0f.txt", r2_res, r3_res, r1, r2, r3) ;
    freopen(name,"w",stdout);
    cout << "#r2_res " << r2_res << endl ;
    cout << "#r3_res " << r3_res << endl ;
    cout << "#r0 " << r0 << endl ;
    cout << "#r1 " << r1 << endl ;
    cout << "#r2 " << r2 << endl ;
    cout << "#r3 " << r3 << endl ;
    cout << "#t " << t << endl ;
    cout << "#ome " << ome << endl ;
    cout << "#adm " << adm << endl ;
    cout << "#komar " << komar << endl ;

    cout << "dom x y z conf lapse bet1 bet2 bet3" << endl ;
    for (int dom=0 ; dom<ndom ; dom++) {
        // Loop on the colocation point
        Index pp (space.get_domain(dom)->get_nbr_points()) ;
        Val_domain xx (space.get_domain(dom)->get_cart(1)) ;
        Val_domain yy (space.get_domain(dom)->get_cart(2)) ;
        Val_domain zz (space.get_domain(dom)->get_cart(3)) ;
        Val_domain con (conf(dom)) ;
        Val_domain lap (lapse(dom)) ;
        Val_domain shif1 (shift(1)(dom)) ;
        Val_domain shif2 (shift(2)(dom)) ;
        Val_domain shif3 (shift(3)(dom)) ;

        do  {
            cout << dom << " "
                 << xx (pp) << " "
                 << yy (pp) << " "
                 << zz (pp) << " "
                 << con (pp) << " "
                 << lap (pp) << " "
                 << shif1 (pp) << " "
                 << shif2 (pp) << " "
                 << shif3 (pp) << endl ;

        } while (pp.inc()) ;
    }

    return EXIT_SUCCESS ;
}



#include <cstdio>
#include "kadath_spheric.hpp"

using namespace Kadath;

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "File missing..." << endl;
        abort();
    }

    int r2_res, r3_res;
    double r0, r1, r2, r3, t, ome, freq, mu, lambda;

    char *name_fich = argv[1];
    FILE *fich = fopen(name_fich, "r");

    Space_spheric space(fich);
    fread_be(&r2_res, sizeof(int), 1, fich);
    fread_be(&r3_res, sizeof(int), 1, fich);
    fread_be(&r0, sizeof(double), 1, fich);
    fread_be(&r1, sizeof(double), 1, fich);
    fread_be(&r2, sizeof(double), 1, fich);
    fread_be(&r3, sizeof(double), 1, fich);
    fread_be(&t, sizeof(double), 1, fich);
    fread_be(&ome, sizeof(double), 1, fich);
    fread_be(&freq, sizeof(double), 1, fich);
    fread_be(&mu, sizeof(double), 1, fich);
    fread_be(&lambda, sizeof(double), 1, fich);
    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    Scalar field1(space, fich);
    Scalar field2(space, fich);
    fclose(fich);

    cout << "reading success" << endl;

    // get number of domains
    int ndom = space.get_nbr_domains();


    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);
    System_of_eqs syst(space, 1, ndom - 1);

    // Unknown
    syst.add_var("ph1", field1);
    syst.add_var("ph2", field2);
    syst.add_var("Psi", conf);
    syst.add_var("N", lapse);
    syst.add_var("bet", shift);

    syst.add_cst("mu", mu);
    syst.add_cst("frq", freq);
    syst.add_cst("lmd", lambda);

    // Metric :
    fmet.set_system(syst, "f");

    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - (2. / 3.)* D_k bet^k * f^ij) /(2.* N)");
    syst.add_def("PI1 = (bet^i * D_i ph1 - frq * ph2) / (2*N)");
    syst.add_def("PI2 = (bet^i * D_i ph2 + frq * ph1) / (2*N)");
    syst.add_def(
            "rho = 2 * (PI1^2 + PI2^2) + mu^2 * (ph1^2 + ph2^2) / 2 + lmd * (ph1^4 + 2 * ph1^2 * ph2^2 + ph2^4) / 8 + (D_i ph1 * D^i ph1 + D_i ph2 * D^i ph2) / (2 * Psi^4)");
    syst.add_def("mom^i = 2 * (PI1 * D^i ph1 + PI2 * D^i ph2) / Psi^4");

    Tensor aij(syst.give_val_def("A"));

    // compute Jadm
    Val_domain a13(aij(1, 3)(ndom - 1));
    Val_domain integ_j(space.get_domain(ndom - 1)->mult_r(a13.mult_cos_theta()));
    double jadm = space.get_domain(ndom - 1)->integ(integ_j, OUTER_BC) / 8 / M_PI;

    cout << "Jadm: " << jadm << endl;


    // Computation adm mass :
    Val_domain integ_adm(conf(ndom - 1).der_r());
    double adm = -space.get_domain(ndom - 1)->integ(integ_adm, OUTER_BC) / 2 / M_PI;

    cout << "adm mass: " << adm << endl;

    // Computation of Komar :
    Val_domain integ_komar(lapse(ndom - 1).der_r());
    double komar = space.get_domain(ndom - 1)->integ(integ_komar, OUTER_BC) / 4 / M_PI;

    cout << "Komar mass: " << komar << endl;


    // Retrieve energy density
    Scalar E(syst.give_val_def("rho"));

    // Retrieve momentum density
    Vector M(syst.give_val_def("mom"));


    char name[100];
    sprintf(name, "backreaction_2p_A%.4f_mu%.2f_lmd%.4f.txt", A, mu, lambda);
    freopen(name, "w", stdout);
    cout << "#r2_res " << r2_res << endl;
    cout << "#r3_res " << r3_res << endl;
    cout << "#r0 " << r0 << endl;
    cout << "#r1 " << r1 << endl;
    cout << "#r2 " << r2 << endl;
    cout << "#r3 " << r3 << endl;
    cout << "#t " << t << endl;
    cout << "#ome " << ome << endl;
    cout << "#freq " << freq << endl;
    cout << "#mu " << mu << endl;
    cout << "#lambda " << lambda << endl;
    cout << "#adm " << adm << endl;
    cout << "#komar " << komar << endl;
    cout << "#Jadm " << jadm << endl;

    cout << "dom x y z r conf lapse bet1 bet2 bet3 field1 field2 rho mom1 mom2 mom3" << endl;
    for (int dom = 1; dom < (ndom - 1); dom++) {
        // Loop on the colocation point
        Index pp(space.get_domain(dom)->get_nbr_points());
        Val_domain xx(space.get_domain(dom)->get_cart(1));
        Val_domain yy(space.get_domain(dom)->get_cart(2));
        Val_domain zz(space.get_domain(dom)->get_cart(3));
        Val_domain rr(space.get_domain(dom)->get_radius());
        Val_domain con(conf(dom));
        Val_domain lap(lapse(dom));
        Val_domain shif1(shift(1)(dom));
        Val_domain shif2(shift(2)(dom));
        Val_domain shif3(shift(3)(dom));
        Val_domain fiel1(field1(dom));
        Val_domain fiel2(field2(dom));
        Val_domain EE(E(dom));
        Val_domain M1(M(1)(dom));
        Val_domain M2(M(2)(dom));
        Val_domain M3(M(3)(dom));
        do {
            cout << dom << " "
                 << xx(pp) << " "
                 << yy(pp) << " "
                 << zz(pp) << " "
                 << rr(pp) << " "
                 << con(pp) << " "
                 << lap(pp) << " "
                 << shif1(pp) << " "
                 << shif2(pp) << " "
                 << shif3(pp) << " "
                 << fiel1(pp) << " "
                 << fiel2(pp) << " "
                 << EE(pp) << " "
                 << M1(pp) << " "
                 << M2(pp) << " "
                 << M3(pp) << endl;

        } while (pp.inc());
    }

    return EXIT_SUCCESS;
}


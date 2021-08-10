#include "kadath_spheric.hpp"
#include "mpi.h"

using namespace Kadath;
using namespace std::map;

int main(int argc, char **argv) {

    int rc = MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        cout << "File missing..." << endl;
        abort();
    }

    // set scalar field amplitude and mass
    double A = 1;     //0.0003 ;
    double mu = 0.18;
    double lambda = 0.0;

    int r2_res, r3_res;
    double r0, r1, r2, r3;
    double t, ome;

    // lookup eigen-frequency corresponding to mu
    std::map<double, double> freq_dict = {{0.10, 0.0994719732994934},
                                          {0.11, 0.1092953566202027},
                                          {0.12, 0.1190736173193454},
                                          {0.13, 0.1288049783744057},
                                          {0.14, 0.138482137126494},
                                          {0.15, 0.1480961731994996},
                                          {0.16, 0.157636041452291},
                                          {0.17, 0.1670882997930869},
                                          {0.18, 0.1764379644998751},
                                          {0.19, 0.185670425087694},
                                          {0.20, 0.1944546186467369},
    };

    // extract eigen-frequency from dictionary
    double freq = freq_dict[mu];

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

    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    fclose(fich);

    if (rank == 0) {
        cout << "cloud 2p" << endl;
        cout << "reading success" << endl;
        cout << "r2_res= " << r2_res << endl;
        cout << "r3_res= " << r3_res << endl;
        cout << "r0 = " << r0 << endl;
        cout << "r1 = " << r1 << endl;
        cout << "r2 = " << r2 << endl;
        cout << "t  = " << t << endl;
        cout << "ome= " << ome << endl;
        cout << "mu = " << mu << endl;
        cout << "frq= " << freq << endl;
        cout << "A  = " << A << endl;
        cout << "lambda = " << lambda << endl;
    }

    int ndom = space.get_nbr_domains();
    if (rank == 0) { cout << "ndom " << ndom << endl; }


    // Computation adm mass :
    Val_domain integ_adm(conf(ndom - 1).der_r());
    double adm = -space.get_domain(ndom - 1)->integ(integ_adm, OUTER_BC) / 2 / M_PI;
    if (rank == 0) { cout << "adm mass: " << adm << endl; }

    double a = adm * mu;
    if (rank == 0) {
        cout << "alpha: " << a << endl;
        cout << "freq: " << freq << endl;
    }

    int r = r0;

    Scalar field1(space);
    field1.set_domain(0) = 0;

    Scalar field2(space);
    field2.set_domain(0) = 0;

    for (int d = 1; d <= (ndom - 1); d++) {
        Val_domain rr = space.get_domain(d)->get_radius();
        rr.std_base_r_spher();
        Val_domain rh = rr + 1 / (16 * rr) + 0.5; // convert to harmonic coordinates
        field1.set_domain(d) = A * rh.mult_cos_phi().mult_sin_theta();
        field2.set_domain(d) = A * rh.mult_sin_phi().mult_sin_theta();
    }

    field1.set_domain(ndom - 1) = 0.;
    field1.set_domain(ndom - 1) = 0.;
    field1.std_base();
    field2.std_base();

    Scalar p1 = field1;
    Scalar p2 = field2;

    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);
    System_of_eqs syst(space, 2, 3);

    // Unknown
    syst.add_var("ph1", field1);
    syst.add_var("ph2", field2);

    syst.add_cst("p1", p1);
    syst.add_cst("p2", p2);

    // constants
    syst.add_cst("mu", mu);
    syst.add_cst("lmd", lambda);
    syst.add_cst("Psi", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("frq", freq);

    // Metric :
    fmet.set_system(syst, "f");

    // Definition of the extrinsic curvature (\tilde{A} in Gourgolhon notes)
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - (2. / 3.)* D_k bet^k * f^ij) /(2.* N)"); // checked

    // definitions of momenta
    syst.add_def("PI1 = (bet^i * D_i ph1 - frq * ph2) / (2*N)"); // checked
    syst.add_def("PI2 = (bet^i * D_i ph2 + frq * ph1) / (2*N)"); // checked

    // conformal second derivative term
    syst.add_def(
            "DP1 = - bet^i * bet^j * D_i D_j ph1 + 2 * bet^i * bet^j * ( D_i Psi * D_j ph1 / Psi + D_j Psi * D_i ph1 / Psi ) - 2 * bet_i * bet^i * D^k Psi * D_k ph1 / Psi"); // checked
    syst.add_def(
            "DP2 = - bet^i * bet^j * D_i D_j ph2 + 2 * bet^i * bet^j * ( D_i Psi * D_j ph2 / Psi + D_j Psi * D_i ph2 / Psi ) - 2 * bet_i * bet^i * D^k Psi * D_k ph2 / Psi"); // checked

    // Lie derivatives of momenta
    syst.add_def(
            "LP1 = (bet^i * D_i ph1 * bet^j * D_j N)/2 - (frq * ph2 * bet^i * D_i N)/2 + (N * frq * bet^i * D_i ph2) + (N * frq^2 * ph1)/2 + N * DP1 / 2 - (N * bet^i * D_i bet^j * D_j ph1)/2"); // checked
    syst.add_def(
            "LP2 = (bet^i * D_i ph2 * bet^j * D_j N)/2 + (frq * ph1 * bet^i * D_i N)/2 - (N * frq * bet^i * D_i ph1) + (N * frq^2 * ph2)/2 + N * DP2 / 2 - (N * bet^i * D_i bet^j * D_j ph2)/2"); // checked

    // definitions of the matter equations
    syst.add_def(
            "rho = 2 * (PI1^2 + PI2^2) + mu^2 * (ph1^2 + ph2^2) / 2 + lmd * (ph1^4 + 2 * ph1^2 * ph2^2 + ph2^4) / 8 + (D_i ph1 * D^i ph1 + D_i ph2 * D^i ph2) / (2 * Psi^4)"); // checked
    syst.add_def("mom^i = 2 * (PI1 * D^i ph1 + PI2 * D^i ph2) / Psi^4") // checked
    syst.add_def(
            "RpS = 8 * (PI1^2 + PI2^2) - mu^2 * (ph1^2 + ph2^2) - lmd * (ph1^4 + 2 * ph1^2 * ph2^2 + ph2^4) / 4") // checked

    // KG equations
    syst.add_def(
            "KG1 = N^2 * D_i N * D^i ph1 / Psi^4 + N^3 * D_i D^i ph1 / Psi^4 + 2 * N^3 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - N^3 * mu * mu * ph1 - N^3 * lmd * (ph1^2 + ph2^2) * ph1 / 2"); // checked
    syst.add_def(
            "KG2 = N^2 * D_i N * D^i ph2 / Psi^4 + N^3 * D_i D^i ph2 / Psi^4 + 2 * N^3 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - N^3 * mu * mu * ph2 - N^3 * lmd * (ph1^2 + ph2^2) * ph2 / 2"); // checked


    // Inner BC sphere minus
    space.add_inner_bc(syst, "ph1=p1");
    space.add_inner_bc(syst, "ph2=p2");

    // Outer BC
    space.add_outer_bc(syst, "ph1=0.");
    space.add_outer_bc(syst, "ph2=0.");

    // KG equations
    space.add_eq(syst, "KG1 = 0.", "ph1", "dn(ph1)");
    space.add_eq(syst, "KG2 = 0.", "ph2", "dn(ph2)");


    bool endloop = false;
    int ite = 1;
    double conv;

    while (!endloop) {
        endloop = syst.do_newton(1e-8, conv);
        if (rank == 0) {
            cout << "Newton iteration " << ite << " " << conv << endl;
        }
        ite++;
    }

    // Output
    if (rank == 0) {
        char name[100];
        sprintf(name, "schwField_2p_A%.4f_mu%.2f.dat", A, mu);
        FILE *fich = fopen(name, "w");
        space.save(fich);
        fwrite_be(&r2_res, sizeof(int), 1, fich);
        fwrite_be(&r3_res, sizeof(int), 1, fich);
        fwrite_be(&r0, sizeof(double), 1, fich);
        fwrite_be(&r1, sizeof(double), 1, fich);
        fwrite_be(&r2, sizeof(double), 1, fich);
        fwrite_be(&r3, sizeof(double), 1, fich);
        fwrite_be(&t, sizeof(double), 1, fich);
        fwrite_be(&ome, sizeof(double), 1, fich);
        fwrite_be(&freq, sizeof(double), 1, fich);
        fwrite_be(&mu, sizeof(double), 1, fich);
        fwrite_be(&lambda, sizeof(double), 1, fich);
        conf.save(fich);
        lapse.save(fich);
        shift.save(fich);
        field1.save(fich);
        field2.save(fich);
        fclose(fich);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

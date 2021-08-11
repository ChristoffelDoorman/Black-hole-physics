#include "kadath_spheric.hpp"
#include "mpi.h"

using namespace Kadath;

int main(int argc, char **argv) {

    int rc = MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        cout << "File missing..." << endl;
        abort();
    }

    int r2_res, r3_res;
    double r0, r1, r2, r3;
    double t, ome;

    // field values
    double mu, A, freq, lambda;

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
    fread_be(&A, sizeof(double), 1, fich);
    fread_be(&lambda, sizeof(double), 1, fich);

    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    Scalar field1(space, fich);
    Scalar field2(space, fich);
    fclose(fich);

    if (rank == 0) {
        cout << "Backreaction for cloud 2p with N and Psi only" << endl;
        cout << "reading success" << endl;
        cout << "r2_res= " << r2_res << endl;
        cout << "r3_res= " << r3_res << endl;
        cout << "r0 = " << r0 << endl;
        cout << "r1 = " << r1 << endl;
        cout << "r2 = " << r2 << endl;
        cout << "r3 = " << r3 << endl;
        cout << "t  = " << t << endl;
        cout << "ome= " << ome << endl;
        cout << "mu = " << mu << endl;
        cout << "A  = " << A << endl;
        cout << "frq= " << freq << endl;
        cout << "lmd= " << lambda << endl;
    }

    int ndom = space.get_nbr_domains();
    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);

    // Computation adm mass before backreaction :
    Val_domain integ_adm_init(conf(ndom - 1).der_r());
    double adm_init = -space.get_domain(ndom - 1)->integ(integ_adm_init, OUTER_BC) / 2 / M_PI;

    // Vector parallel to the sphere (needed only for inner BC)
    Vector mm(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        mm.set(i) = 0.;
    Val_domain xx(space.get_domain(1)->get_cart(1));
    Val_domain yy(space.get_domain(1)->get_cart(2));
    mm.set(3).set_domain(1) = sqrt(xx * xx + yy * yy);
    mm.std_base();

    // Normal to sphere BH1 :
    Vector n(space, COV, basis);
    n.set(1) = 1.;
    n.set(2) = 0.;
    n.set(3) = 0.;
    n.std_base();

    // copy fields for inner boundary values
    Scalar p1 = field1;
    Scalar p2 = field2;

    field1.set_domain(1).annule_hard();
    field2.set_domain(1).annule_hard();

    System_of_eqs syst(space, 1, ndom - 1);

    // Metric :
    fmet.set_system(syst, "f");

    // fields to solve
    syst.add_var("Psi", conf);
    syst.add_var("N", lapse);

    // One user defined constant
    syst.add_cst ("ph1", field1) ;
    syst.add_cst ("ph2", field2) ;
    syst.add_cst ("bet", shift) ;
    syst.add_cst("lmd", lambda);
    syst.add_cst("r", r0);
    syst.add_cst("t", t);
    syst.add_cst("n", n);
    syst.add_cst("ome", ome);
    syst.add_cst("m", mm);
    syst.add_cst("mu", mu);
    syst.add_cst("frq", freq);
    syst.add_cst("pi", M_PI);
    // inner boundary values (double)
    syst.add_cst("p1", p1);
    syst.add_cst("p2", p2);

    // normalized normal to BH sphere
    syst.add_def("nn^i = n^i / sqrt(n_i * n^i)"); // checked

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
    syst.add_def("mom^i = 2 * (PI1 * D^i ph1 + PI2 * D^i ph2) / Psi^4"); // checked
    syst.add_def(
            "RpS = 8 * (PI1^2 + PI2^2) - mu^2 * (ph1^2 + ph2^2) - lmd * (ph1^4 + 2 * ph1^2 * ph2^2 + ph2^4) / 4"); // checked

    // KG equations
    syst.add_def(
            "KG1 = N^2 * D_i N * D^i ph1 / Psi^4 + N^3 * D_i D^i ph1 / Psi^4 + 2 * N^3 * D_i ph1 * D^i Psi / Psi^5 + 2 * LP1 - N^3 * mu * mu * ph1 - N^3 * lmd * (ph1^2 + ph2^2) * ph1 / 2"); // checked
    syst.add_def(
            "KG2 = N^2 * D_i N * D^i ph2 / Psi^4 + N^3 * D_i D^i ph2 / Psi^4 + 2 * N^3 * D_i ph2 * D^i Psi / Psi^5 + 2 * LP2 - N^3 * mu * mu * ph2 - N^3 * lmd * (ph1^2 + ph2^2) * ph2 / 2"); // checked

    // Field equations
    for (int d = 1; d < ndom; d++) {
        if ((d != 2) && (d != 3)) {
            syst.add_def(d, "eqN = D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij"); // checked
            syst.add_def(d, "eqP = D_i D^i Psi + Psi^5 *A_ij * A^ij / 8"); // checked
            syst.add_def(d,
                         "eqB^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2 * A^ij * (D_j N - 6 * N * D_j Psi / Psi)"); // checked
        } else {
            syst.add_def(d,
                         "eqN = D_i D^i N + 2 * D_i Psi * D^i N / Psi - N * Psi^4 * A_ij *A^ij - 4 * pi * N * Psi^4 * RpS"); // checked
            syst.add_def(d, "eqP = D_i D^i Psi + Psi^5 * A_ij * A^ij / 8 + 2 * pi * rho * Psi^5"); // checked
            syst.add_def(d,
                         "eqB^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2 * A^ij * (D_j N - 6 * N * D_j Psi / Psi) - 16 * pi * N * Psi^4 * mom^i"); // checked
        }
    }

    // inner BC metric
    space.add_inner_bc(syst, "N = t"); // checked
//    space.add_inner_bc(syst, "bet^i = t / Psi^2 * nn^i + ome * m^i * r"); // checked
    space.add_inner_bc(syst, "4 * nn^i * D_i Psi / Psi + D_i nn^i + Psi^2 * A_ij * nn^i * nn^j = 0"); // checked

    // outer BC metric
    space.add_outer_bc(syst, "N = 1"); // checked
    space.add_outer_bc(syst, "Psi = 1"); // checked
//    space.add_outer_bc(syst, "bet^i = 0"); // checked

    // Einstein Equations
    space.add_eq(syst, "eqN = 0", "N", "dn(N)"); // checked
    space.add_eq(syst, "eqP = 0", "Psi", "dn(Psi)"); // checked
//    space.add_eq(syst, "eqB^i= 0", "bet^i", "dn(bet^i)"); // checked

    // For the fields :
    // O near the horizon
//    syst.add_eq_full(1, "ph1=0");
//    syst.add_eq_full(1, "ph2=0");

    // Inner BC cst value
//    syst.add_eq_bc(2, INNER_BC, "ph1 = p1");
//    syst.add_eq_bc(2, INNER_BC, "ph2 = p2");

    // comment out if r_mid
//    for (int d = 2; d <= 3; d++) {
//        syst.add_eq_inside(d, "KG1=0");
//        syst.add_eq_inside(d, "KG2=0");
//
//        // Matching
//        if (d != 3) {
//            syst.add_eq_matching(2, OUTER_BC, "ph1");
//            syst.add_eq_matching(2, OUTER_BC, "dn(ph1)");
//            syst.add_eq_matching(2, OUTER_BC, "ph2");
//            syst.add_eq_matching(2, OUTER_BC, "dn(ph2)");
//        }
//    }

    //Outer BC
//    syst.add_eq_bc(3, OUTER_BC, "ph1 = 0.");
//    syst.add_eq_bc(3, OUTER_BC, "ph2 = 0.");

    // 0 elsewhere
//    for (int d = 4; d < ndom; d++) {
//        syst.add_eq_full(d, "ph1=0");
//        syst.add_eq_full(d, "ph2=0");
//    }

    if (rank == 0) {
        Array<double> errors(syst.check_equations());
        for (int i = 0; i < errors.get_size(0); i++)
            if (errors(i) > 1)
                cout << i << " " << errors(i) << endl;
    }

    // Newton-Raphson
    double conv;
    bool endloop = false;
    int ite = 1;
    if (rank == 0)
        cout << "Solve all fields simultaneously" << endl;
    while (!endloop) {
        endloop = syst.do_newton(1e-8, conv);
        if (rank == 0) {
            cout << "Newton iteration " << ite << " " << conv << endl;
            ite++;

            // Output interim results
            char name[100];
            sprintf(name, "br_N_Psi_A%.4f_mu%.2f_lmbd%.4f.dat", A, mu, lambda);
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
            fwrite_be(&A, sizeof(double), 1, fich);
            fwrite_be(&lambda, sizeof(double), 1, fich);
            conf.save(fich);
            lapse.save(fich);
            shift.save(fich);
            field1.save(fich);
            field2.save(fich);
            fclose(fich);
        }
    }

    // Computation adm mass after backreaction :
    Val_domain integ_adm(conf(ndom - 1).der_r());
    double adm = -space.get_domain(ndom - 1)->integ(integ_adm, OUTER_BC) / 2 / M_PI;
    if (rank == 0) {
        cout << "ADM mass before backreaction: " << adm_init << endl;
        cout << "ADM mass after backreaction:  " << adm << endl;
    }

    // Output final results
    if (rank == 0) {
        char name[100];
        sprintf(name, "br_N_Psi_A%.4f_mu%.2f_lmbd%.4f.dat", A, mu, lambda);
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
        fwrite_be(&A, sizeof(double), 1, fich);
        fwrite_be(&lambda, sizeof(double), 1, fich);
        conf.save(fich);
        lapse.save(fich);
        shift.save(fich);
        field1.save(fich);
        field2.save(fich);
        fclose(fich);
    }
    MPI_Finalize();
}

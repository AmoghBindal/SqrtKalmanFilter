import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=12, suppress=True)


def make_problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    P_shape = B @ B.T / n + np.eye(n)
    H = rng.standard_normal((m, n))
    C = rng.standard_normal((m, m))
    R = C @ C.T / m + np.eye(m)
    return P_shape, H, R


def reference_update(P_pred, H, R):
    P, H, R = (x.astype(np.float64) for x in (P_pred, H, R))
    S = H @ P @ H.T + R
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(P.shape[0]) - K @ H
    return A @ P @ A.T + K @ R @ K.T


def kalman_updates(P_pred, H, R):
    n = P_pred.shape[0]
    I = np.eye(n, dtype=P_pred.dtype)

    # Standard and Joseph
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S, H @ P_pred).T
    A = I - K @ H
    P_standard = A @ P_pred
    P_joseph = A @ P_pred @ A.T + K @ R @ K.T

    # Square root: QR generalizes the scalar hypot() of the 1-D version
    L_pred = np.linalg.cholesky(P_pred)
    L_R = np.linalg.cholesky(R)

    # Innovation factor: S = L_S L_S^T
    _, U_S = np.linalg.qr(np.vstack([(H @ L_pred).T, L_R.T]))
    L_S = U_S.T

    # K = P H^T S^{-1}, computed from the factors
    PHt = L_pred @ (H @ L_pred).T
    K_sqrt = np.linalg.solve(L_S.T, np.linalg.solve(L_S, PHt.T)).T
    A_sqrt = I - K_sqrt @ H

    # Factorized Joseph: P = [A L_pred, K L_R][A L_pred, K L_R]^T
    _, U = np.linalg.qr(np.vstack([(A_sqrt @ L_pred).T, (K_sqrt @ L_R).T]))
    L_updated = U.T
    P_sqrt = L_updated @ L_updated.T  # reconstructed only for comparison

    return P_standard, P_joseph, P_sqrt


def errors_vs_reference(P_list, P_ref):
    return [np.linalg.norm(P.astype(np.float64) - P_ref) for P in P_list]


def compare_filters(dtype, n=4, m=2, p0=1e8):
    print(f"\n{'=' * 65}")
    print(f"Testing with {dtype.__name__}  (n={n}, m={m}, P_pred scale={p0:g})")
    print("=" * 65)

    P_shape, H, R = make_problem(n, m)
    P_pred = (p0 * P_shape).astype(dtype)
    H, R = H.astype(dtype), R.astype(dtype)

    P_standard, P_joseph, P_sqrt = kalman_updates(P_pred, H, R)
    P_ref = reference_update(P_pred, H, R)

    for name, P in [("Standard", P_standard), ("Joseph", P_joseph), ("Square root", P_sqrt)]:
        asym = np.linalg.norm(P - P.T)
        min_eig = np.linalg.eigvalsh((P + P.T) / 2).min()
        print(f"\n{name} covariance:\n{P}")
        print(f"  asymmetry ||P - P^T||: {asym:.3g}   min eigenvalue: {min_eig:.3g}")

    print("\nFrobenius-norm errors vs float64 Joseph reference:")
    for name, e in zip(["Standard", "Joseph", "Square root"],
                       errors_vs_reference([P_standard, P_joseph, P_sqrt], P_ref)):
        print(f"  {name:12s} {e:.12g}")


def plot_error_comparison(dtype, n=4, m=2):
    exponents = np.arange(0, 30)
    P_shape, H, R = make_problem(n, m)
    H, R = H.astype(dtype), R.astype(dtype)

    errors = {"Standard": [], "Joseph": [], "Square root": []}
    print(f"\nError sweep ({dtype.__name__}, n={n}, m={m})")
    print(f"{'scale':>8} {'Standard':>14} {'Joseph':>14} {'Square root':>14}")
    for exp in exponents:
        P_pred = (10.0 ** exp * P_shape).astype(dtype)
        P_ref = reference_update(P_pred, H, R)
        results = errors_vs_reference(kalman_updates(P_pred, H, R), P_ref)
        for name, e in zip(errors, results):
            errors[name].append(e)
        print(f"{'1e' + str(exp):>8} " + " ".join(f"{e:14.4e}" for e in results))

    plt.figure(figsize=(9, 6))
    for (name, errs), marker in zip(errors.items(), ["o-", "s-", "^-"]):
        plt.semilogy(exponents, errs, marker, label=f"{name} update")
    plt.xlabel(r"$\log_{10}$(scale of $P_{\mathrm{pred}}$)")
    plt.ylabel(r"$\|P - P_{\mathrm{ref}}\|_F$")
    plt.title(f"Covariance update error, n={n}, m={m} ({dtype.__name__})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n, m = 10, 10

    compare_filters(np.float32, n, m)
    plot_error_comparison(np.float32, n, m)

    # compare_filters(np.float64, n, m)
    # plot_error_comparison(np.float64, n, m)

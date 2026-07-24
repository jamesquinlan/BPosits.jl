using BPosits

# Lorenz attractor via explicit Euler integration.
# Demonstrates how arithmetic precision affects trajectory stability.
# Change T to observe precision effects.

T = BPosit32

sigma = T(10);  rho = T(28);  beta = T(8) / T(3)
dt    = T(0.01)
nsteps = 10_000

function lorenz_step(x, y, z, sigma, rho, beta, dt)
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return x + dt*dx, y + dt*dy, z + dt*dz
end

function simulate(x0, y0, z0, sigma, rho, beta, dt, nsteps)
    xs = Vector{typeof(x0)}(undef, nsteps)
    ys = Vector{typeof(y0)}(undef, nsteps)
    zs = Vector{typeof(z0)}(undef, nsteps)
    x, y, z = x0, y0, z0
    for i in 1:nsteps
        isnan(x) || isnan(y) || isnan(z) && (println("NaR at step $i"); break)
        x, y, z = lorenz_step(x, y, z, sigma, rho, beta, dt)
        xs[i] = x;  ys[i] = y;  zs[i] = z
    end
    xs, ys, zs
end

xs, ys, zs = simulate(T(1), T(1), T(1), sigma, rho, beta, dt, nsteps)
println("Final point: (", Float64(xs[end]), ", ", Float64(ys[end]), ", ", Float64(zs[end]), ")")

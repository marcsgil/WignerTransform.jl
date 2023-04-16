module WignerTransform

using FastTransforms,Interpolations

export wigner_transform

symetric_range(N) = range(-N÷2,length=N)
interval(grid) = grid[2] - grid[1]
reciprocal_interval(grid) = π/last(grid)
direct_grid(grid) = interval(grid) * symetric_range(length(grid))
reciprocal_grid(grid) = reciprocal_interval(grid) * symetric_range(length(grid))

function wigner_transform(ψ, qs, ps, plan)
    auto_correlation = map( Q -> ψ(Q[2]+Q[1]/2)*conj(ψ(Q[2]-Q[1]/2)), Iterators.product(reciprocal_grid(ps),qs) )

	real(fftshift(plan * ifftshift(auto_correlation,1),1))/(2*last(ps))
end

function wigner_transform(ψ::Union{Interpolations.AbstractInterpolation,Function}, qs, ps)
    wigner_transform(ψ, qs, ps, plan_fft(Array{complex(eltype(ps))}(undef,length(ps),length(qs)),1))
end

function wigner_transform(ψ::AbstractArray{T,1}, qs, ps) where T
    wigner_transform(cubic_spline_interpolation(qs, ψ, extrapolation_bc=0), qs, ps)
end

function wigner_transform(ψs::AbstractArray{T,2}, qs, ps) where T
    plan = plan_fft(view(ψs,:,:,1),1)
    [ wigner_transform(cubic_spline_interpolation(qs, ψ, extrapolation_bc=0), qs, ps, plan) for ψ ∈ eachslice(ψs,dims=2)] |> stack
end


end

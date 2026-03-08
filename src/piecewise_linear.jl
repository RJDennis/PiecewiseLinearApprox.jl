"""
piecewise_linear_nodes(n, domain = [1.0, -1.0])

Generate `n` evenly-spaced nodes over `domain`.

# Arguments
- `n::Integer`: number of nodes; must be positive.
- `domain`: two-element vector `[ub, lb]` defining the interval.

# Returns
A `Vector` of `n` node locations sorted from smallest to largest.

# Examples
```julia
piecewise_linear_nodes(5)          # [-1.0, -0.5, 0.0, 0.5, 1.0]
piecewise_linear_nodes(3, [0.0, 1.0])  # [0.0, 0.5, 1.0]
```
"""
function piecewise_linear_nodes(n::S,domain = [1.0,-1.0]) where {S <: Integer}

  if n <= 0
    error("The number of nodes must be positive.")
  end

  nodes = [(domain[1]+domain[2])/2.0 for _ in 1:n]

  if isodd(n)
    inc = (domain[1]-domain[2])/(n-1)
  else
    inc = (domain[1]-domain[2])/n
  end
  @inbounds for i = 1:div(n,2)
    nodes[i]     += (i-1-div(n,2))*inc
    nodes[n-i+1] -= (i-1-div(n,2))*inc
  end

  return nodes

end

"""
linear_nodes(n, domain = [1.0, -1.0])

Alias for [`piecewise_linear_nodes`](@ref).
"""
const linear_nodes = piecewise_linear_nodes

function bracket_nodes(x::AbstractVector{T}, point::R) where {T<:AbstractFloat, R<:Number}

    realpoint = real(point) # Real is used because complex numbers are occasionally used in NLboxsolve.jl
    n = length(x)

    realpoint <= x[1]   && return (1,2)
    realpoint >= x[n] && return (n-1,n)
    y = searchsortedlast(x, realpoint) 
    return (y, y+1)

end

function bracket_nodes(x::Union{NTuple{d,Array{T,1}},Array{Array{T,1},1}},point::AbstractVector{R}) where {T <: AbstractFloat, R <: Number, d}

  bracketing_nodes = Array{Int64,2}(undef,2,length(point))

  @inbounds for i in eachindex(point)
    bracketing_nodes[:,i] .= bracket_nodes(x[i],point[i])
  end

  return bracketing_nodes

end

function piecewise_linear_weight(x::AbstractVector{T},point::R) where {T<:AbstractFloat,R<:Number}

    bracketing_nodes = bracket_nodes(x,point)

    weight = (point-x[bracketing_nodes[1]])/(x[bracketing_nodes[2]]-x[bracketing_nodes[1]])

    return weight

end

function piecewise_linear_weight(x::AbstractVector{T}, point::R, brackets::AbstractVector{S}) where {T <: AbstractFloat, R <: Number, S <: Integer}

    w = (point - x[brackets[2]]) / (x[brackets[1]] - x[brackets[2]])

    return w

end

function piecewise_linear_weights(x::Union{NTuple{d,Array{T,1}},Array{Array{T,1},1}},point::AbstractVector{R}) where {T <: AbstractFloat, R <: Number, d}

    weights = Array{R,1}(undef,length(point))

    @inbounds for i in eachindex(point)
        weights[i] = piecewise_linear_weight(x[i],point[i])
    end

    return weights

end

function piecewise_linear_weights(x::Union{NTuple{d,Array{T,1}},Array{Array{T,1},1}},point::AbstractVector{R},brackets::AbstractMatrix{S}) where {T <: AbstractFloat, R <: Number, S <: Integer, d}

    weights = Array{R,1}(undef,length(point))

    @inbounds for i in eachindex(point)
        weights[i] = piecewise_linear_weight(x[i],point[i],brackets[:,i])
    end

    return weights

end

function select_bracketing_nodes(bounds::Array{S,2}) where {S <: Integer}

  d = size(bounds,2)

  bracketing_grid_points = Array{S,2}(undef,2^d,d)

  @inbounds for i = 1:d
    @views bracketing_grid_points[:,i] .= repeat(repeat(bounds[:,i],inner = 2^(d-i)),inner = 2^(i-1))
  end

  return bracketing_grid_points

end

function select_relevant_data(y::AbstractArray{T,d}, bracketing_grid_points::Array{S,2}) where {T <: AbstractFloat, S<: Integer, d}

    data = Vector{T}(undef, 2^d)
    @inbounds for i in eachindex(data)
        data[i] = y[CartesianIndex(ntuple(j -> bracketing_grid_points[i,j], d))]
    end

    return data

end

"""
piecewise_linear_evaluate(y, x, point)
piecewise_linear_evaluate(y, x)

Evaluate a piecewise linear interpolant defined by function values `y` on grid
`x` at `point`.  The multi-dimensional interpolant is constructed by successive linear
interpolation (multilinear interpolation) along each dimension.

# Arguments
- `y`: array of function values on the grid. For an N-dimensional problem `y`
  is an N-dimensional array whose size along dimension `i` equals
  `length(x[i])`.
- `x`: grid nodes — either an `NTuple` or a `Vector` of 1-D node vectors, one
  per dimension. Use [`piecewise_linear_nodes`](@ref) to generate them.
- `point`: the query location — a scalar (1-D) or a vector of length N (N-D).

When called with only `y` and `x`, returns a closure `f(point)` that evaluates
the interpolant at any `point`.

# Returns
The interpolated scalar value at `point`, or a callable when `point` is
omitted.

# Examples
```julia
x = piecewise_linear_nodes(5)
y = sin.(x)
piecewise_linear_evaluate(y, x, 0.3)   # interpolated sin(0.3)

f = piecewise_linear_evaluate(y, x)
f(0.3)                                  # same result via closure
```
"""
function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}}) where {T <: AbstractFloat, R <: Number, N}

  b = bracket_nodes(x,point)
  w = piecewise_linear_weights(x,point,b)

  d = size(b,2)

  relevant_points = select_bracketing_nodes(b)
  data = select_relevant_data(y,relevant_points)

  @inbounds for j = d:-1:1

    @inbounds for i in 1:div(2^j,2)
      data[i] = data[2*(i-1)+1] + w[j]*(data[2*i]-data[2*(i-1)+1])
    end

    data = @view data[1:div(2^j,2)]

  end

  return data[1]

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}}) where {T <: AbstractFloat, N}

  function approximating_function(point::Union{R,Array{R,1}}) where {R<:Number}

    return piecewise_linear_evaluate(y,x,point)

  end

  return approximating_function

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}},integrals::Union{T,Array{T,1}}) where {T <: AbstractFloat, R <: Number, N}

  # This function is only needed to facilitate compatibility with SolveDSGE

  b = bracket_nodes(x,point)
  w = piecewise_linear_weights(x,point,b)
  #w = w.*integrals

  d = size(b,2)

  relevant_points = select_bracketing_nodes(b)
  data = select_relevant_data(y,relevant_points)

  @inbounds for j = d:-1:1

    for i in 1:div(2^j,2)
      data[i] = data[2*(i-1)+1] + w[j]*integrals[j]*(data[2*i]-data[2*(i-1)+1])
    end

    data = @view data[1:div(2^j,2)]

  end

  return data[1]

end

# Functions to handle the 1-D case

function piecewise_linear_evaluate(y::Array{T,1},x::Array{T,1},point::R) where {T <: AbstractFloat,R <: Number}

    b = bracket_nodes(x,point)
    w = piecewise_linear_weight(x,point,b)

    y_estimate = y[b[1]] + w*(y[b[2]]-y[b[1]])

    return y_estimate

end

function piecewise_linear_evaluate(y::Array{T,1},nodes::Array{T,1}) where {T <: AbstractFloat}

  function approximating_function(point::R) where {R<:Number}

      return piecewise_linear_evaluate(y,nodes,point)

  end

  return approximating_function

end

# Function to transform from one uniform grid to another

function grid_reshape(f::AbstractArray{T,N},grid::NTuple{N,Array{T,1}}) where {T <: AbstractFloat, N}

  #1. Construct the old grid
    
  old_grid = Array{Array{T,1},1}(undef,N)
  @inbounds for i = 1:N
    old_grid[i] = piecewise_linear_nodes(size(f,i),[grid[i][end],grid[i][1]])
  end

  #2. Initialize the new y

  y = Array{T,N}(undef,tuple(length.(grid)...))

  #3. Interpolate to fill y

  point = Array{T,1}(undef,N)
  @inbounds for i in CartesianIndices(y)
    for j = 1:N
      point[j] = grid[j][i[j]]
    end
    y[i] = piecewise_linear_evaluate(f,old_grid,point)
  end

  return y

end

"""
piecewise_linear_derivative(y, x, point, pos)

Compute the partial derivative of a piecewise linear interpolant with respect
to dimension `pos` at `point` using a central finite difference.

The derivative is approximated as

    (f(point + h·eₚₒₛ) - f(point - h·eₚₒₛ)) / (2h)

with `h = 1e-2`, where `eₚₒₛ` is the unit vector along dimension `pos`.

# Arguments
- `y`: N-dimensional array of function values on the grid.
- `x`: grid nodes — `NTuple` or `Vector` of 1-D node vectors.
- `point`: query location as a vector of length N.
- `pos::Integer`: index of the dimension to differentiate with respect to.

# Returns
The scalar finite-difference approximation of the partial derivative.

# Examples
```julia
x1 = piecewise_linear_nodes(10, [0.0, 1.0])
x2 = piecewise_linear_nodes(10, [0.0, 1.0])
y  = [sin(a + b) for a in x1, b in x2]
piecewise_linear_derivative(y, [x1, x2], [0.5, 0.5], 1)  # ∂/∂x₁
```
"""
function piecewise_linear_derivative(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}},pos::S) where {S <: Integer, T <: AbstractFloat, R <: Number, N}

  h = 1e-2

  point_upper = copy(point)
  point_lower = copy(point)

  point_upper[pos] += h
  point_lower[pos] -= h

  y_estimate_upper = piecewise_linear_evaluate(y,x,point_upper)
  y_estimate_lower = piecewise_linear_evaluate(y,x,point_lower)

  deriv = (y_estimate_upper -  y_estimate_lower)/(2*h)

  return deriv

end

function piecewise_linear_derivative(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}},integrals::Union{T,Array{T,1}},pos::S) where {S <: Integer, T <: AbstractFloat, R <: Number, N}

  # This function is only needed to facilitate compatibility with SolveDSGE
  
  h = 1e-2

  point_upper = copy(point)
  point_lower = copy(point)

  point_upper[pos] += h
  point_lower[pos] -= h

  y_estimate_upper = piecewise_linear_evaluate(y,x,point_upper,integrals)
  y_estimate_lower = piecewise_linear_evaluate(y,x,point_lower,integrals)

  deriv = (y_estimate_upper -  y_estimate_lower)/(2*h)

  return deriv

end
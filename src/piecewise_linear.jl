#=

Code to implement piecewise linear interpolation for an arbitrary number
of dimensions

=#

function piecewise_linear_nodes(n::S,domain = [1.0,-1.0]) where {S <: Integer}

  if n <= 0
    error("The number of nodes must be positive.")
  end

  nodes = zeros(n)

  if isodd(n)
    nodes[Int((n-1)/2)+1] = (domain[1]+domain[2])/2.0
  end

  if n == 1

    return nodes
    
  elseif n == 2
    
    return [domain[2], domain[1]]
    
  else
    
    nodes[1] = domain[2]
    nodes[n] = domain[1]

    inc = (domain[1]-domain[2])/(n-1)
    for i = 2:div(n,2)
      nodes[i]     = domain[2] + (i-1)*inc
      nodes[n-i+1] = domain[1] - (i-1)*inc
    end

    return nodes
    
  end

end

const linear_nodes = piecewise_linear_nodes

function bracket_nodes(x::Array{T,1},point::R) where {T <: AbstractFloat,R<:Number}

  if real(point) <= x[1] # Real is used because complex numbers are occasionally used in NLboxsolve.jl
    return (1,2)
  elseif real(point) >= x[end]
    return (length(x) - 1, length(x))
  else
    y = 0
    for i in x
      if i < real(point)
        y += 1
      else
        break
      end
    end
    return (y, y+1)
  end

end
 
function bracket_nodes(x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Array{R,1}) where {T <: AbstractFloat, N,R<:Number}

  bracketing_nodes = Array{Int64,2}(undef,2,length(point))
  for i = 1:length(point)
    bracketing_nodes[:,i] .= bracket_nodes(x[i],point[i])
  end

  return bracketing_nodes

end

function piecewise_linear_weight(x::Array{T,1},point::R) where {T<:AbstractFloat,R<:Number}

    bracketing_nodes = bracket_nodes(x,point)

    weight = (point-x[bracketing_nodes[1]])/(x[bracketing_nodes[2]]-x[bracketing_nodes[1]])

    return weight

end

function piecewise_linear_weights(x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Array{R,1}) where {T<:AbstractFloat,N,R<:Number}

    weights = Array{R,1}(undef,length(point))
    for i = 1:length(point)
        weights[i] = piecewise_linear_weight(x[i],point[i])
    end

    return weights

end

function select_bracketing_nodes(bounds::Array{S,2} where {S <: Integer})

  d = size(bounds,2)

  number_relevant_grid_points = 2^d
  bracketing_grid_points      = Array{Int64}(undef,number_relevant_grid_points,d)

  for i = 1:d
    bracketing_grid_points[:,i] .= repeat(repeat(bounds[:,i],inner=[2^(d-i),1]),2^(i-1))[:]
  end

  return bracketing_grid_points

end

function select_relevant_data(y::AbstractArray{T,N},bracketing_grid_points::Array{S,2}) where {T <: AbstractFloat, S <: Integer, N}

  data = zeros(size(bracketing_grid_points,1))
  for i = 1:length(data)
    data[i] = y[CartesianIndex(tuple(bracketing_grid_points[i,:]...))]
  end

  return data

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}}) where {T<:AbstractFloat,N,R<:Number}

  b = bracket_nodes(x,point)
  w = piecewise_linear_weights(x,point)

  d = size(b,2)

  relevant_points = select_bracketing_nodes(b)
  data = select_relevant_data(y,relevant_points)

  for j = d:-1:1

    new_data = zeros(R,div(length(data),2))
    for i = 1:length(new_data)
      new_data[i] = data[2*(i-1)+1] + w[j]*(data[2*i]-data[2*(i-1)+1])
    end

    data = copy(new_data)

  end

  return data[1]

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}}) where {T <: AbstractFloat, N}

  function approximating_function(point::Union{R,Array{R,1}}) where {R<:Number}

    return piecewise_linear_evaluate(y,x,point)

  end

  return approximating_function

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{R,AbstractArray{R,1}},integrals::Union{T,Array{T,1}}) where {T<:AbstractFloat,N,R<:Number}

  # This function is only needed to facilitate compatibility with SolveDSGE

  b = bracket_nodes(x,point)
  w = piecewise_linear_weights(x,point)
  w = w.*integrals

  d = size(b,2)

  relevant_points = select_bracketing_nodes(b)
  data = select_relevant_data(y,relevant_points)

  for j = d:-1:1

    new_data = zeros(R,Int(length(data)/2))
    for i = 1:length(new_data)
      new_data[i] = data[2*(i-1)+1] + w[j]*(data[2*i]-data[2*(i-1)+1])
    end

    data = copy(new_data)

  end

  return data[1]

end

# Functions to handle the 1-D case

function piecewise_linear_evaluate(y::Array{T,1},x::Array{T,1},point::R) where {T<:AbstractFloat,R<:Number}

    b = bracket_nodes(x,point)
    w = piecewise_linear_weight(x,point)

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

function grid_reshape(f::AbstractArray{T,N},grid::NTuple{N,Array{T,1}}) where {T<:AbstractFloat,N}

  #1. Construct the old grid
    
  old_grid = Array{Array{T,1},1}(undef,N)
  for i = 1:N
    piecewise_linear_nodes
    old_grid[i] = piecewise_linear_nodes(size(f,i),[grid[i][end],grid[i][1]])
  end

  #2. Initialize the new y

  y = Array{T,N}(undef,tuple(length.(grid)...))

  #3. Interpolate to fill y

  point = Array{T,1}(undef,N)
  for i in CartesianIndices(y)
    for j = 1:N
      point[j] = grid[j][i[j]]
    end
    y[i] = piecewise_linear_evaluate(f,old_grid,point)
  end

  return y

end

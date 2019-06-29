#=

Code to implement piecewise linear interpolation for an arbitrary number
of dimensions

=#

function piecewise_linear_nodes(n::S,domain = [1.0,-1.0]) where {S <: Integer}

    nodes = collect(range(domain[2],domain[1],length=n)) # The nodes are ordered from lowest to highest
    return nodes

end

const linear_nodes = piecewise_linear_nodes

function bracket_nodes(x::Array{T,1},point::T) where {T <: AbstractFloat}

    if point <= x[1]
        return (1,2)
    elseif point >= x[end]
        return (length(x) - 1, length(x))
    else
        return (sum(point .> x), sum(point .> x) + 1)
    end

end

function bracket_nodes(x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Array{T,1}) where {T <: AbstractFloat, N}

    bracketing_nodes = zeros(Int64,2,length(x))
    for i = 1:length(point)
        bracketing_nodes[:,i] .= bracket_nodes(x[i],point[i])
    end
    return bracketing_nodes
end

function piecewise_linear_weight(x::Array{T,1},point::T) where {T <: AbstractFloat}

    bracketing_nodes = bracket_nodes(x,point)
    weight = (point-x[bracketing_nodes[1]])/(x[bracketing_nodes[2]]-x[bracketing_nodes[1]])

    return weight

end

function piecewise_linear_weights(x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Array{T,1}) where {T <: AbstractFloat, N}

    weights = Array{T}(undef,length(point))
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

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{T,Array{T,1}}) where {T <: AbstractFloat, N}

  b = bracket_nodes(x,point)
  w = piecewise_linear_weights(x,point)

  d = size(b,2)

  relevant_points = select_bracketing_nodes(b)
  data = select_relevant_data(y,relevant_points)

  for j = d:-1:1

    new_data = zeros(Int(length(data)/2))
    for i = 1:length(new_data)
      new_data[i] = data[2*(i-1)+1] + w[j]*(data[2*i]-data[2*(i-1)+1])
    end

    data = copy(new_data)

  end

   return data[1]

end

function piecewise_linear_evaluate(y::AbstractArray{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}}) where {T <: AbstractFloat, N}

  function approximating_function(point::Union{T,Array{T,1}}) where {T <: AbstractFloat}

    return piecewise_linear_evaluate(y,x,point)

  end

  return approximating_function

end

#=

function locate_point_below(x::Array{T,1},point::T) where {T <: AbstractFloat}

    if point <= x[1]
        return 1
    elseif point >= x[end]
        return length(x) - 1
    else
        return sum(point .> x)
    end

end

function piecewise_linear_evaluate(y::Array{T,N},x::Union{NTuple{N,Array{T,1}},Array{Array{T,1},1}},point::Union{T,Array{T,1}}) where {T <: AbstractFloat, N}

    if length(x) == 1
        x_below = locate_point_below(x[1],point[1])
        x_above = x_below + 1
        y_hat = y[x_below] + ((point[1] - x[1][x_below])/(x[1][x_above] - x[1][x_below]))*(y[x_above] - y[x_below])
        return y_hat
    else
        x_below = locate_point_below(x[end],point[end])
        x_above = x_below + 1
        yy = zeros(size(y)[1:end-1])
        for i in CartesianIndices(yy)
            yy[i] = y[CartesianIndex(i,x_below)] + ((point[end] - x[end][x_below])/(x[end][x_above] - x[end][x_below]))*(y[CartesianIndex(i,x_above)] - y[CartesianIndex(i,x_below)])
        end
        x = x[1:end-1]
        point = point[1:end-1]
        piecewise_linear_evaluate(yy,x,point)
    end

end

=#

function piecewise_linear_evaluate(y::Array{T,1},nodes::Array{T,1},point::T) where {T <: AbstractFloat}

    y_estimate = piecewise_linear_evaluate(y,(nodes,),point)

    return y_estimate

end

function piecewise_linear_evaluate(y::Array{T,1},nodes::Array{T,1}) where {T <: AbstractFloat}

  function approximating_function(point::T) where {T <: AbstractFloat}

      return piecewise_linear_evaluate(y,nodes,point)

  end

  return approximating_function

end

#=

Code to implement piecewise linear interpolation for an arbitrary number
of dimensions

=#

function piecewise_linear_nodes(n::S,domain = [one(T),-one(T)]) where {T <: AbstractFloat, S <: Integer}

    nodes = collect(range(domain[2],domain[1],length=n)) # The nodes are ordered from lowest to highest
    return nodes

end

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

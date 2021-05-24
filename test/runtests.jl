using PiecewiseLinearApprox
using Test

@testset "PiecewiseLinearApprox.jl" begin

  function foo(x)

    y = log.(x[1])*log.(x[2])'
    return y

  end

  x1 = piecewise_linear_nodes(11,[3.0,1.5])
  x2 = piecewise_linear_nodes(21,[2.5,0.5])

  x = (x1,x2)
  y = foo(x)

  point = [2.1,0.7]

  y_hat = piecewise_linear_evaluate(y,x,point)

  y_actual = log(point[1])*log(point[2])
  diff = (y_actual - y_hat)
  println(diff)
  @test abs(diff) <= 1e-3
  
end

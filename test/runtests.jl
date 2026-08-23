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
  
  function piecewise_linear_tests(;verbose::Bool = true)

    pass = true
    report(name,ok) = (verbose && println(rpad(name,52), ok ? "pass" : "FAIL"); ok)

    # nodes are evenly spaced for every n, odd or even
    ok = true
    for n = 2:9
      nd = piecewise_linear_nodes(n,[1.0,-1.0])
      sp = diff(nd)
      ok &= maximum(abs.(sp .- sp[1])) < 1e-12
      ok &= isapprox(nd[1],-1.0) && isapprox(nd[end],1.0)
    end
    pass &= report("nodes evenly spaced, n = 2 to 9",ok)

    # the bracketing cell has 2^d distinct corners
    ok = true
    for d = 1:4
      bounds = vcat(fill(1,1,d),fill(2,1,d))
      ok &= length(unique(eachrow(select_bracketing_nodes(bounds)))) == 2^d
    end
    pass &= report("select_bracketing_nodes gives 2^d corners",ok)

    # the 1-D path returns a value rather than raising
    x = piecewise_linear_nodes(11,[1.0,-1.0])
    y = 2.0 .* x .+ 1.0
    ok = isapprox(piecewise_linear_evaluate(y,x,0.3),2*0.3+1,atol = 1e-12)
    pass &= report("1-D evaluate, linear data reproduced",ok)

    # exactness: asymmetric multilinear functions, a different grid per dimension
    x1 = piecewise_linear_nodes(3,[1.0,0.0])
    x2 = piecewise_linear_nodes(4,[2.0,0.0])
    x3 = piecewise_linear_nodes(5,[3.0,0.0])

    f2(a,b)   = 1 + 2a + 3b + 4a*b
    f3(a,b,c) = 1 + 2a + 3b + 5c + 7a*b + 11a*c + 13b*c + 17a*b*c

    g2 = [f2(a,b) for a in x1, b in x2]
    g3 = [f3(a,b,c) for a in x1, b in x2, c in x3]

    e2 = maximum(abs(piecewise_linear_evaluate(g2,(x1,x2),[a,b]) - f2(a,b))
                 for (a,b) in ((0.1,0.9),(0.25,1.4),(0.83,0.2),(0.4,1.75)))
    pass &= report("2-D exact on a bilinear function",e2 < 1e-12)

    e3 = maximum(abs(piecewise_linear_evaluate(g3,(x1,x2,x3),[a,b,c]) - f3(a,b,c))
                 for (a,b,c) in ((0.1,0.9,2.2),(0.7,1.6,0.4),(0.35,0.25,1.1)))
    pass &= report("3-D exact on a trilinear function",e3 < 1e-12)
 
    # Second-order convergence on smooth, non-multilinear data.  The error is
    # measured at the CELL MIDPOINTS of whatever grid is in use, which is where a
    # multilinear interpolant is least accurate.  Measuring instead at a fixed set
    # of points does not show the order cleanly -- those points sit at different
    # relative positions in their cells as the grid is refined, and the ratios come
    # out erratic (3.4, 16.1, 2.4, 2.7 in one such attempt) even though the scheme
    # is second order.  At the midpoints the ratio per doubling is 3.35, 3.63,
    # 3.80, 3.90, converging on the 4 that h^2 requires.
    errs = Float64[]
    for n in (11,41,161)
      xa = piecewise_linear_nodes(n,[3.0,1.5])
      xb = piecewise_linear_nodes(n,[2.5,0.5])
      ga = log.(xa)*log.(xb)'
      ma = [(xa[i]+xa[i+1])/2 for i = 1:n-1]
      mb = [(xb[j]+xb[j+1])/2 for j = 1:n-1]
      push!(errs,maximum(abs(piecewise_linear_evaluate(ga,(xa,xb),[a,b]) - log(a)*log(b))
                         for a in ma, b in mb))
    end
    pass &= report("second-order convergence (ratios > 10)",
                   errs[1]/errs[2] > 10 && errs[2]/errs[3] > 10)

    # the derivative is the exact slope of the cell
    xd = piecewise_linear_nodes(21,[1.0,0.0])
    yd = [3a + 5b for a in xd, b in xd]
    ok = isapprox(piecewise_linear_derivative(yd,(xd,xd),[0.42,0.31],1),3.0,atol = 1e-10) &&
         isapprox(piecewise_linear_derivative(yd,(xd,xd),[0.42,0.31],2),5.0,atol = 1e-10)
    pass &= report("derivative exact on a linear function",ok)

    verbose && println(rpad("OVERALL",52), pass ? "pass" : "FAIL")

    return pass

  end
end

# PLS: Partial Least Squares Regression

Partial Least Squares or PLS is a regression technique which is used for predicting a response $Y$ using a set of predictors $X$ using a small number of latent components. The latent components are linear combinition of predictors which can be regressed on instead of the original set of regressors $X$. This is particularly helpful if we have a large number of correlated predictors. In this scenario, oridinary least squares fails whereas PLS chooses its components to capture the directions of $X$ that are most predictive of $Y$.   

In this documentation, we will depomstrate implementation of Plskern using `BigRiverEssence.plskern` on a `gasoline` dataset. 


## The method

Let $Y$ be response variable and $X$ be a matrix of regressors. PLS finds a direction $w$ (one as a time) in the predictor space such that it maximizes the covariance between the projected predictors $Xw$ and the response $Y$. It considers the following optimization problem: $$\max_{w} \; \operatorname{cov}(Xw, Y) \quad \text{subject to } \|w\|_2 = 1.$$

The scores $t = Xw$ are the values of the samples along that component. The predictor
and response loadings ($p$ and $q$) are used for describing how $X$ and $Y$ are related to the scores. We then deflate or remove the component from the cross covariance. This enables the next component to captures new predictive structure rather than repeating what has been found. After we obtain $k$ components, prediction reduces to a linear model $\hat Y = \text{intercept} +
X B$, where the coefficient matrix $B$ is assembled from the accumulated weights and loadings.


`plskern` of `BigRiverEssence` implements the improved kernel algorithms used by Dayal & MacGregor (1997) where they use  cross-product matrices for efficient computation. The number of components, `nlv`, is the main tuning parameter of `plskern` where 
a small value leads underfitting and vice versa.


## The data

The gasoline dataset (near-infrared spectra and octane numbers for 60 gasoline samples)
is from Kalivas (1997) [1], obtained via the R `pls` package. The NIR spectra were
measured as log(1/R) from 900 to 1700 nm in 2 nm steps, giving 401 wavelengths.



```@example pmd
using BigRiverEssence, DelimitedFiles, Plots, Statistics, Random
```


```@example pmd



datadir = joinpath(pkgdir(BigRiverEssence), "reference_Data", "gasolinedata")
NIR    = readdlm(joinpath(datadir, "NIR.csv"),    ',', Float64)   # 60 × 401 spectra
octane = vec(readdlm(joinpath(datadir, "octane.csv"), ',', Float64))  # 60 responses
wl     = collect(900:2:1700)         # the known wavelength grid (nm)
size(NIR), length(octane)
```

## Train/test split

Now, since PLS is a predictive model, we consider a hold out test set. We use a random seed for reproduction.




```@example pmd
Random.seed!(42)
idx = shuffle(1:size(NIR, 1))
train, test = idx[1:40], idx[41:end]

Xtr = Matrix{Float64}(NIR[train, :]); ytr = reshape(Float64.(octane[train]), :, 1)
Xte = Matrix{Float64}(NIR[test, :]);  yte = reshape(Float64.(octane[test]),  :, 1)

```

It is important to keep in mind that `plskern` takes the
response as a matrix which gives it flexibility to handles more than one responses uniformly. Hence we reshaped `ytr` and `yte` to $n\times 1$ matrices. 


## Fitting the model `plskern`

We now fit a PLS model with $5$ latent components.



```@example pmd
m = plskern(Xtr, ytr; nlv = 5)
```

The function returns the full PLS factorization: the weights `W` and
`R`, the predictor and response loadings `P` and `Q`, the scores `T`, and the
centering means and scales.


```@example pmd
@show size(m.W)
@show size(m.P)
@show size(m.Q)
@show size(m.T)
```


## Predicting with `plskern_predict`

We now predict octane for the held-out samples in the test set and compare with the measured values.



```@example pmd
ŷ = plskern_predict(m, Xte)

rmse = sqrt(mean((vec(ŷ) .- vec(yte)).^2))     # test-set prediction error
@show rmse
scatter(vec(yte), vec(ŷ); legend = false,
    xlabel = "measured octane", ylabel = "predicted octane",
    title = "PLS prediction on held-out samples")
lims = extrema(vcat(vec(yte), vec(ŷ)))
plot!(collect(lims), collect(lims); color = :black, linestyle = :dash)
```


We see from the above plot that the predicted octane values fall veryclose to the dashed $45°$ line of perfect prediction. We also get a test-set RMSE of about 0.1822 octane units. This shows a good calibration from spectra alone, on samples the model never saw during fitting.


## The regression coefficients with `plskern_coef`

The function `plskern_coef` is used for assembling the coefficient vector $B$ and intercept that map a raw
spectrum directly to a predicted octane. We can  plot $B$ against wavelength to see which
spectral regions drive the prediction.




```@example pmd
B, intercept = plskern_coef(m)

plot(wl, vec(B); legend = false,
    xlabel = "wavelength (nm)", ylabel = "regression coefficient",
    title = "PLS coefficients across the NIR spectrum")
hline!([0], color = :black, alpha = 0.3)
```

we see from the plot that most wavelengths carry small coefficients, with few high peaks where the spectral
regions are most informative about octane. This is the interpretable payoff of a linear
calibration where the coefficients point back to the chemistry. 

## Projecting with `plskern_transform`

The other function `plskern_transform` is used for projecting samples onto the PLS latent space or their scores. If we color
the scores by octane, it will shows that the components are organized around the response. This is the
supervised counterpart to the variance-only components of PCA.


```@example pmd
Ttr = plskern_transform(m, Xtr)

scatter(Ttr[:, 1], Ttr[:, 2]; zcolor = vec(ytr), colorbar = true, legend = false,
    xlabel = "PLS score 1", ylabel = "PLS score 2",
    title = "Samples in PLS latent space (colored by octane)")
```

After coloring the score scatter by octane, in the above plot, we get to know what makes PLS supervised. We see the samples
lining up along the first component by their octane value where dark (low-octane) points to
the left and bright (high-octane) to the right. The first PLS component is giving the direction
of the spectra most predictive of octane, not merely the direction of greatest
spectral variance as in PCA. That is why regressing on a few PLS scores captures the
response so efficiently.

## Summary

We saw that from 60 spectra with 401 correlated predictors, `plskern` were able to built a handful of latent
components that predict octane accurately on held-out samples. This is the setting where PLS
outperforms ordinary regression. 

`plskern` is most suitable in regression problems where there are huge number of correlated variables where ordinary least squares. In this example, we used the `gasoline` dataset which has $401$ variables but only $60$ samples. `plskern` can be used in any similar type of datasets.

## References

[1] Kalivas, J. H. (1997). Two data sets of near infrared spectra. *Chemometrics and
    Intelligent Laboratory Systems*, 37, 255–259.

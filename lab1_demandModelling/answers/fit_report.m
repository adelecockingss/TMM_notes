function fit_report(name, Tm, Te)
%FIT_REPORT Print CPC, correlation and RMSE of model Tm against observed Te.
R = corrcoef(Tm(:), Te(:));
fprintf('%-32s CPC = %.3f   r = %.3f   RMSE = %.1f\n', name, cpc(Tm, Te), R(1,2), ...
        sqrt(mean((Tm(:) - Te(:)).^2)));
end

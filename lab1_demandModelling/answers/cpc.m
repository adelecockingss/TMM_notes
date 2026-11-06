function s = cpc(Tm, Te)
%CPC Sorensen index / common part of commuters, 0 (no overlap) to 1 (perfect).
%   CPC = 2 sum min(Tm, Te) / (sum Tm + sum Te)
s = 2*sum(min(Tm, Te), 'all')/(sum(Tm, 'all') + sum(Te, 'all'));
end

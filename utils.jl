function evaluate_method(Hs, mtp_result)
    rj_idx = mtp_result.rjs_idx
    discoveries = sum(rj_idx)
    true_discoveries = sum(rj_idx .& Hs)
    false_discoveries = discoveries - true_discoveries
    FDP = false_discoveries / max(discoveries, 1)
    Power = true_discoveries / max( sum(Hs), 1)

    (FDP = FDP, Power=Power, discoveries = discoveries,)
end
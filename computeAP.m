function ap = computeAP(label, score, gt)
    rand_index = randperm(length(label));
    label2 = label(rand_index);
    score = score(rand_index);
    [~, sids] = sort(score, 'descend');
    label2 = label2(sids);
    ids = find(label2 == gt);
    ap = 0;
    for j = 1:length(ids)
        ap  = ap + j / (ids(j) * length(ids));
    end
    fprintf('%f \n', ap);
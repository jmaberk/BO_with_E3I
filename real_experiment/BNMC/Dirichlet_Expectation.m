function myexpectation=Dirichlet_Expectation(alpha)
    if size(alpha,1)==1
        myexpectation=psi(alpha)-psi(sum(alpha));
    else
        myexpectation=bsxfun(@minus,psi(alpha),psi(sum(alpha,2)));
    end
end
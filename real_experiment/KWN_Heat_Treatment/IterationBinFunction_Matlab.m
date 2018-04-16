function [Binprecippercum_totalbin,steprad,numprecippercum_totalbin]=IterationBinFunction_Matlab(totalbinnumber,dissolutionsize,steprad,radsurfenchange,...
    fsurfen,isurfen,msurfen,xsolvus_cookstepcount,timestep,Binprecippercum_totalbin,...
    xarconst,tempk,Diff,xmatrix_totalbin,numberofnucleations,xp,Binprecippercumconst)

% only consider nonzero entry
%idxNonZero=steprad(steprad>dissolutionsize);
idxNonZero=find(steprad>=dissolutionsize);
steprad_nonzero=steprad(idxNonZero);
idxNotFsurfen=find(steprad_nonzero<radsurfenchange);
mysurfen=fsurfen*ones(1,length(idxNonZero));
mysurfen(idxNotFsurfen)=(msurfen*steprad_nonzero(idxNotFsurfen)+isurfen);

temp=xarconst/tempk;
%xar=xsolvus_cookstepcount*exp(xarconst*mysurfen./(tempk*steprad_nonzero));
xar=xsolvus_cookstepcount*exp(temp*mysurfen./steprad_nonzero);

steprad_nonzero=steprad_nonzero+timestep*(Diff.*(xmatrix_totalbin-xar))./((xp-xar).*steprad_nonzero);
steprad(idxNonZero)=steprad_nonzero; % put back to steprad

Binprecippercum_totalbin=Binprecippercum_totalbin+sum((steprad_nonzero.*steprad_nonzero.*steprad_nonzero)*Binprecippercumconst.*numberofnucleations(idxNonZero));

numprecippercum_totalbin=sum(numberofnucleations(idxNonZero));
return;

Iteratingbinnumber=1;

while Iteratingbinnumber<totalbinnumber
    
    steprad_iter=steprad(Iteratingbinnumber);
    if steprad_iter>=dissolutionsize
        
        %if radius of precipitates nucleated at time
        %Iteratingbinnumber is physically possible, the precipitate
        %will grow/shrink depending on Gibbs-Thomson relations
        
        if steprad_iter>=radsurfenchange
            surfen=fsurfen;
        else
            surfen=(msurfen*steprad_iter+isurfen);
        end
        
        %xar=xsolvus_cookstepcount*exp(2*surfen*molvol/(gas*tempk*steprad(Iteratingbinnumber)));
        xar=xsolvus_cookstepcount*exp(xarconst*surfen/(tempk*steprad_iter));
        
        %the effective equilibrium composition at edge of
        %precipitates in this bin, accounting for gibbs-thomson
        %steprad(Iteratingbinnumber)=steprad(Iteratingbinnumber)+timestep*(Diff*(xmatrix(totalbinnumber)-xar))/((xp-xar)*steprad(Iteratingbinnumber));
        steprad_iter=steprad_iter+timestep*(Diff*(xmatrix_totalbin-xar))/((xp-xar)*steprad_iter);
        %calculated radius of precipitates in this bin after
        %the current timestep
        %numprecippercum(totalbinnumber) = numprecippercum(totalbinnumber)+numberofnucleations(Iteratingbinnumber);
        %numprecippercum_totalbin = numprecippercum_totalbin+numberofnucleations(Iteratingbinnumber);
    else
        %dissolve precipitates if they are below the minimal
        %physical precipitate size.
        steprad_iter=0;
    end
    
   
    Binprecippercum_totalbin=Binprecippercum_totalbin+(steprad_iter*steprad_iter*steprad_iter)*Binprecippercumconst*numberofnucleations(Iteratingbinnumber);
    
    steprad(Iteratingbinnumber)=steprad_iter;
        
    %look at the next historical timestep and loop
    Iteratingbinnumber=Iteratingbinnumber+1;
end

end
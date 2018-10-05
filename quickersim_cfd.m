[p,e,t] = importMeshGmsh('cylinder.msh');

[p,e,t,nVnodes, nPnodes,indices] = convertMeshToSecondOrder(p,e,t);
Re = 1500;
nu = 1/Re;

u = initSolution(p, t, [1,0],0);

for i = 1:105
    [NS, F] = assembleNavierStokesMatrix2D(p,e,t,nu,u(indices.indu),u(indices.indv), 'nosupg');
    
    [NS,F] = imposeCfdBoundaryCondition2D(p,e,t,NS,F,1, 'inlet', [1,0]);
    [NS,F] = imposeCfdBoundaryCondition2D(p,e,t,NS,F,3,'wall');
    [NS,F] = imposeCfdBoundaryCondition2D(p,e,t,NS,F,4,'wall');
    
    u = NS\F;
end

displaySolution2D(p,t,u(indices.indu),'x-velocity');
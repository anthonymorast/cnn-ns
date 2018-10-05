%%% -------------------------------------------------- %%%
%%% Author: Denys Dutykh, CNRS -- LAMA, Univ of Savoie %%%
%%% E-mail: Denys.Dutykh@univ-savoie.fr                %%%
%%% Web:    http://www.denys-dutykh.com/               %%%
%%% Blog:   http://dutykh.github.io/                   %%%
%%% GitHub: https://github.com/dutykh/                 %%%
%%% -------------------------------------------------- %%%

function Plot (Om, t, ~)

    global dy Lx Ly X Y Re
    fn = ['imgs/',num2str(Re)];
    if ~exist(fn, 'dir')
        mkdir(fn);
    end
    %save(['data/soln',num2str(t,'%4.2f'), '_', num2str(Re, '%d'), '.mat'], "Om");

    surf(X, Y, Om); 
    grid off;
    shading interp;
    colormap(jet); 
    cc = colorbar;
    xlim([-Lx Lx]); ylim([-Ly+dy Ly]); caxis([-1 1]); zlim([-1.2, 1.2]);
    xlabel('$x$', 'interpreter', 'latex', 'fontsize', 12);
    ylabel('$y$', 'interpreter', 'latex', 'fontsize', 12, 'Rotation', 1);
    xlabel(cc, '$\omega(x,y,t)$', 'interpreter', 'latex', 'fontsize', 12, 'Rotation', 90);
    
    %view([0 90]);
    
    title (['Vorticity distribution at t = ',num2str(t,'%4.2f')], 'interpreter', 'latex', 'fontsize', 12);
    set(gcf, 'Color', 'w');
    
    % uncomment to remove x and y numbers (for data for CNN)
    %set(gca, 'YTick', []);
    %set(gca, 'XTick', []);
    drawnow
    print(['imgs/',num2str(Re),'/soln',num2str(t,'%4.2f'), '_', num2str(Re, '%d'), '.png'], '-dpng');
end % Plot ()
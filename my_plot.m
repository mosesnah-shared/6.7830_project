%% 6.7830 Project
clear all; close all; clc;
cd( fileparts( matlab.desktop.editor.getActiveFilename ) ); 

%% For Variational Expectation Maximization (vEM)

% The number of documents 
n_doc = [ 50, 100, 200, 300, 400, 500, 600, 700 ];
N = length( n_doc );

data_raw = cell( 1, N );
lgd      = cell( 1, N );

% Define the Dataset 
for i = 1 : N

    M = n_doc( i );
    data_raw{ i } = load( ['./dataset/set2/trained_v', num2str( M ), '_LDA_EM.mat' ] ) ;

end

% Plotting the L function 
f = figure( ); a = axes( 'parent', f );
hold on

tmp_x = zeros( 1, N );
for i = 1: N
   plot( data_raw{ i }.lb_arr )
   lgd{ i } = [ 'M=', num2str( data_raw{ i }.M ), ' V=' num2str( data_raw{ i }.V ) ] ;

   tmp_x( i ) = length( data_raw{ i }.lb_arr );

end

set( a, 'xlim', [ 0, max( tmp_x ) ], 'fontsize', 25 )
legend( lgd, 'location', 'northeastoutside' )
xlabel( 'Iteration', 'fontsize', 34 )

exportgraphics( f, './figures/fig1.pdf','ContentType','vector' )
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

%% Plotting the topic vs iteration

close

% The number of topics 
k_arr = [ 3, 5, 8, 12, 20 ];
N = length( k_arr );

data_raw = cell( 1, N );
lgd      = cell( 1, N );

% Define the Dataset 
for i = 1 : N

    k = k_arr( i );
    data_raw{ i } = load( ['./dataset/set3/trained_v500_', num2str( k ), '_LDA_EM.mat' ] ) ;

end

% Plotting the L function 
f = figure( ); a = axes( 'parent', f );
hold on


for i = 1: N
   plot( data_raw{ i }.lb_arr, 'linewidth', 6 )
   lgd{ i } = [ 'k=', num2str( k_arr( i ) ) ] ;
end

legend( lgd )

set( a, 'xlim', [ 0, 45 ], 'fontsize', 25 )
legend( lgd, 'location', 'northeastoutside' )
xlabel( 'Iteration', 'fontsize', 34 )

exportgraphics( f, './figures/fig2.pdf','ContentType','vector' )

%% Plotting the number of samples vs. performance.

close

% The number of topics 
n_samples = [ 300, 500, 1000, 2000, 3000, 4000 ];
N = length( n_samples );

data_raw = cell( 1, N );
lgd      = cell( 1, N );

% Define the Dataset 
for i = 1 : N

    n = n_samples( i );
    data_raw{ i } = load( ['./dataset/set4//trained_v500_', num2str( n ), '_LDA_Gibbs.mat' ] ) ;

end
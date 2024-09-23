from medical_data_visualizer import draw_cat_plot, draw_heat_map

cat_plot_fig = draw_cat_plot()
cat_plot_fig.savefig('catplot.png')

heat_map_fig = draw_heat_map()
heat_map_fig_savefig ('heatmap.png')

print("Plots have been saved as 'catplot.ng' and 'heat_map.png'.")
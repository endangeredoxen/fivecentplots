import pdb
db = pdb.set_trace
import test_boxplot
import test_contour
import test_grouping
import test_heatmap
import test_hist
import test_plot
import test_ranges
import test_styles
import test_ticks

def make_all():
    test_boxplot.make_all()
    test_contour.make_all()
    test_grouping.make_all()
    test_heatmap.make_all()
    test_hist.make_all()
    test_plot.make_all()
    test_ranges.make_all()
    test_styles.make_all()
    test_ticks.make_all()
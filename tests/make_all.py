import pdb
db = pdb.set_trace
import test_barplot
import test_boxplot
import test_contour
import test_grouping
import test_heatmap
import test_hist
import test_imshow
import test_misc
import test_nq
import test_pie
import test_plot
import test_ranges
import test_styles
import test_ticks

def make_all():
    test_barplot.make_all()
    test_boxplot.make_all()
    test_contour.make_all()
    test_grouping.make_all()
    test_heatmap.make_all()
    test_hist.make_all()
    test_imshow.make_all()
    test_misc.make_all()
    test_nq.make_all()
    test_pie.make_all()
    test_plot.make_all()
    test_ranges.make_all()
    test_styles.make_all()
    test_ticks.make_all()
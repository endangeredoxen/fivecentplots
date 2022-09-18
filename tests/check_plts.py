import pdb
import test_barplot
import test_boxplot
import test_contour
import test_gantt
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
import fivecentplots.utilities as utl
db = pdb.set_trace


def check_all():
    missing = []
    missing += utl.test_checker(test_barplot)
    missing += utl.test_checker(test_boxplot)
    missing += utl.test_checker(test_contour)
    missing += utl.test_checker(test_grouping)
    missing += utl.test_checker(test_heatmap)
    missing += utl.test_checker(test_hist)
    missing += utl.test_checker(test_gantt)
    missing += utl.test_checker(test_imshow)
    missing += utl.test_checker(test_misc)
    missing += utl.test_checker(test_nq)
    missing += utl.test_checker(test_pie)
    missing += utl.test_checker(test_plot)
    missing += utl.test_checker(test_ranges)
    missing += utl.test_checker(test_styles)
    missing += utl.test_checker(test_ticks)
    print(missing)


if __name__ == '__main__':
    check_all()

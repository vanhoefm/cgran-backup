/* -*- c++ -*- */
%include "gnuradio.i"

%{
#include "sigplot_sink_c.h"
%}

GR_SWIG_BLOCK_MAGIC(sigplot,sink_c)

  sigplot_sink_c_sptr sigplot_make_sink_c (int fftsize, int wintype,
				       double fc=0, double bw=1.0,
				       const std::string &name="Display",
				       bool plotfreq=true, bool plotwaterfall=true,
				       bool plotwaterfall3d=true, bool plottime=true,
				       bool plotconst=true,
				       bool use_openGL=true,
				       QWidget *parent=NULL);

class sigplot_sink_c : public gr_block
{
private:
  friend sigplot_sink_c_sptr sigplot_make_sink_c (int fftsize, int wintype,
					      double fc, double bw,
					      const std::string &name,
					      bool plotfreq, bool plotwaterfall,
					      bool plotwaterfall3d, bool plottime,
					      bool plotconst,
					      bool use_openGL,
					      QWidget *parent);
  sigplot_sink_c (int fftsize, int wintype,
		double fc, double bw,
		const std::string &name,
		bool plotfreq, bool plotwaterfall,
		bool plotwaterfall3d, bool plottime,
		bool plotconst,
		bool use_openGL,
		QWidget *parent);

public:
  void exec_();
  PyObject* pyqwidget();

  void set_frequency_range(const double centerfreq,
			   const double bandwidth);
  void set_time_domain_axis(double min, double max);
  void set_constellation_axis(double xmin, double xmax,
			      double ymin, double ymax);
  void set_frequency_axis(double min, double max);
  void set_constellation_pen_size(int size);
};



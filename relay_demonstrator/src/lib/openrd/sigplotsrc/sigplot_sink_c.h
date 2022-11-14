/* -*- c++ -*- */
#ifndef INCLUDED_SIGPLOT_SINK_C_H
#define INCLUDED_SIGPLOT_SINK_C_H

#include <Python.h>
#include <gr_block.h>
#include <gr_firdes.h>
#include <gri_fft.h>
#include <qapplication.h>
#include "sigplot.h"
#include "sigplotguiclass.h"

class sigplot_sink_c;
typedef boost::shared_ptr<sigplot_sink_c> sigplot_sink_c_sptr;

sigplot_sink_c_sptr sigplot_make_sink_c(int fftsize, int wintype,
		double fc=0, double bw=1.0,
		const std::string& name="Spectrum Display",
		bool plotfreq=true, bool plotwaterfall=true,
		bool plotwaterfall3d=true, bool plottime=true,
		bool plotconst=true,
		bool use_openGL=true,
		QWidget* parent=NULL);

class sigplot_sink_c : public gr_block
{
private:
	friend sigplot_sink_c_sptr sigplot_make_sink_c(int fftsize, int wintype,
			double fc, double bw,
			const std::string& name,
			bool plotfreq, bool plotwaterfall,
			bool plotwaterfall3d, bool plottime,
			bool plotconst,
			bool use_openGL,
			QWidget* parent);

	sigplot_sink_c(int fftsize, int wintype,
			double fc, double bw, 
			const std::string& name,
			bool plotfreq, bool plotwaterfall,
			bool plotwaterfall3d, bool plottime,
			bool plotconst,
			bool use_openGL,
			QWidget* parent);

	void forecast(int noutput_items, gr_vector_int& ninput_items_required);

	// use opengl to force OpenGL on or off
	// this might be necessary for sessions over SSH
	void initialize(const bool opengl=true);

	int d_fftsize;
	gr_firdes::win_type d_wintype;
	std::vector<float> d_window;
	double d_center_freq;
	double d_bandwidth;
	std::string d_name;

	pthread_mutex_t d_pmutex;

	bool d_shift;
	gri_fft_complex* d_fft;

	int d_index;
	gr_complex* d_residbuf;

	bool d_plotfreq, d_plotwaterfall, d_plotwaterfall3d, d_plottime, d_plotconst;

	double d_update_time;

	QWidget* d_parent;
	SigplotGUIClass* d_main_gui;

	void windowreset();
	void buildwindow();
	void fftresize();
	void fft(const gr_complex* data_in, int size);

public:
	~sigplot_sink_c();
	void exec_();
	void lock();
	void unlock();
	QWidget* qwidget();
	PyObject* pyqwidget();

	void set_frequency_range(const double centerfreq,
			const double bandwidth);

	void set_time_domain_axis(double min, double max);
	void set_constellation_axis(double xmin, double xmax,
			double ymin, double ymax);
	void set_constellation_pen_size(int size);
	void set_frequency_axis(double min, double max);

	void set_update_time(double t);

	QApplication* d_qApplication;
	sigplot_obj* d_object;

	int general_work(int noutput_items,
			gr_vector_int &ninput_items,
			gr_vector_const_void_star &input_items,
			gr_vector_void_star &output_items);
};

#endif /* INCLUDED_SIGPLOT_SINK_C_H */


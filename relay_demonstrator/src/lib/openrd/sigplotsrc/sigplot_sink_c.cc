/* -*- c++ -*- */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sigplot_sink_c.h"
#include <gr_io_signature.h>
#include <string.h>

#include <QTimer>

sigplot_sink_c_sptr sigplot_make_sink_c(int fftsize, int wintype,
		double fc, double bw, const std::string& name, 
		bool plotfreq, bool plotwaterfall, bool plotwaterfall3d, bool plottime,
		bool plotconst, bool use_openGL, QWidget* parent)
{
	return sigplot_sink_c_sptr(
			new sigplot_sink_c (fftsize, wintype, fc, bw, name, 
				plotfreq, plotwaterfall, plotwaterfall3d, plottime, plotconst,
				use_openGL, parent));
}

sigplot_sink_c::sigplot_sink_c(int fftsize, int wintype,
		double fc, double bw, const std::string& name,
		bool plotfreq, bool plotwaterfall, bool plotwaterfall3d, bool plottime,
		bool plotconst, bool use_openGL, QWidget *parent) : 
	gr_block ("sink_c",
			gr_make_io_signature (1, -1, sizeof(gr_complex)),
			gr_make_io_signature (0, 0, 0)),
	d_fftsize(fftsize),
	d_wintype((gr_firdes::win_type)(wintype)), 
	d_center_freq(fc), d_bandwidth(bw), d_name(name),
	d_plotfreq(plotfreq), d_plotwaterfall(plotwaterfall),
	d_plotwaterfall3d(plotwaterfall3d), d_plottime(plottime),
	d_plotconst(plotconst),
	d_parent(parent)
{
	d_main_gui = NULL;
	pthread_mutex_init(&d_pmutex, NULL);
	lock();

	// Perform fftshift operation;
	// this is usually desired when plotting
	d_shift = true;  

	d_fft = new gri_fft_complex(d_fftsize, true);

	d_index = 0;
	d_residbuf = new gr_complex[d_fftsize];

	buildwindow();

	initialize(use_openGL);
}

sigplot_sink_c::~sigplot_sink_c()
{
	delete d_main_gui;
	delete [] d_residbuf;
	delete d_fft;
}

void sigplot_sink_c::forecast(int noutput_items, gr_vector_int& ninput_items_required)
{
	unsigned int ninputs = ninput_items_required.size();
	for(unsigned int i = 0; i < ninputs; i++) 
	{
		ninput_items_required[i] = std::min(d_fftsize, 8191);
	}
}

void sigplot_sink_c::lock()
{
	pthread_mutex_lock(&d_pmutex);
}

void sigplot_sink_c::unlock()
{
	pthread_mutex_unlock(&d_pmutex);
}


void sigplot_sink_c::initialize(const bool opengl)
{
	if(qApp != NULL) 
	{
		d_qApplication = qApp;
	}
	else 
	{
		int argc;
		char **argv = NULL;
		d_qApplication = new QApplication(argc, argv);
	}

	if(d_center_freq < 0) 
	{
		throw std::runtime_error("qtgui_sink_c: Received bad center frequency.\n");
	}

	uint64_t maxBufferSize = 32768;
	d_main_gui = new SigplotGUIClass(maxBufferSize, d_fftsize, 
			d_center_freq, -d_bandwidth/2.0, d_bandwidth/2.0);

	d_main_gui->SetDisplayTitle(d_name);
	d_main_gui->SetFFTSize(d_fftsize);
	d_main_gui->SetWindowType((int)d_wintype);

	d_main_gui->OpenSpectrumWindow(d_parent, 
			d_plotfreq, d_plotwaterfall, d_plotwaterfall3d, d_plottime, 
			d_plotconst, opengl);

	// initialize update time to 10 times a second
	set_update_time(0.1);

	d_object = new sigplot_obj(d_qApplication);
	qApp->postEvent(d_object, new sigplot_event(&d_pmutex));
}


void sigplot_sink_c::exec_()
{
	d_qApplication->exec();
}

QWidget* sigplot_sink_c::qwidget()
{
	return d_main_gui->qwidget();
}

PyObject* sigplot_sink_c::pyqwidget()
{
	PyObject *w = PyLong_FromVoidPtr((void*)d_main_gui->qwidget());
	PyObject *retarg = Py_BuildValue("N", w);
	return retarg;
}

void sigplot_sink_c::set_frequency_range(const double centerfreq, 
				  const double bandwidth)
{
	d_center_freq = centerfreq;
	d_bandwidth = bandwidth;
	d_main_gui->SetFrequencyRange(d_center_freq, 
			-d_bandwidth/2.0,
			d_bandwidth/2.0);
}

void sigplot_sink_c::set_time_domain_axis(double min, double max)
{
	d_main_gui->SetTimeDomainAxis(min, max);
}

void sigplot_sink_c::set_constellation_axis(double xmin, double xmax,
		double ymin, double ymax)
{
	d_main_gui->SetConstellationAxis(xmin, xmax, ymin, ymax);
}

void sigplot_sink_c::set_constellation_pen_size(int size)
{
	d_main_gui->SetConstellationPenSize(size);
}

void sigplot_sink_c::set_frequency_axis(double min, double max)
{
	d_main_gui->SetFrequencyAxis(min, max);
}

void sigplot_sink_c::set_update_time(double t)
{
	d_update_time = t;
	d_main_gui->SetUpdateTime(d_update_time);
}

void sigplot_sink_c::fft(const gr_complex *data_in, int size)
{
	if (d_window.size()) 
	{
		gr_complex *dst = d_fft->get_inbuf();
		for(int i = 0; i < size; i++)		// apply window
			dst[i] = data_in[i] * d_window[i];
	}
	else 
	{
		memcpy (d_fft->get_inbuf(), data_in, sizeof(gr_complex)*size);
	}

	d_fft->execute ();     // compute the fft
}

void sigplot_sink_c::windowreset()
{
	gr_firdes::win_type newwintype = (gr_firdes::win_type)d_main_gui->GetWindowType();  
	if(d_wintype != newwintype) 
	{
		d_wintype = newwintype;
		buildwindow();
	}
}

void sigplot_sink_c::buildwindow()
{
	d_window.clear();
	if(d_wintype != 0) 
	{
		d_window = gr_firdes::window(d_wintype, d_fftsize, 6.76);
	}
}

void sigplot_sink_c::fftresize()
{
	int newfftsize = d_main_gui->GetFFTSize();

	if(newfftsize != d_fftsize) 
	{
		// Resize residbuf and replace data
		delete [] d_residbuf;
		d_residbuf = new gr_complex[newfftsize];

		// Set new fft size and reset buffer index 
		// (throws away any currently held data, but who cares?) 
		d_fftsize = newfftsize;
		d_index = 0;

		// Reset window to reflect new size
		buildwindow();

		// Reset FFTW plan for new size
		delete d_fft;
		d_fft = new gri_fft_complex (d_fftsize, true);
	}
}


int sigplot_sink_c::general_work (int noutput_items,
		gr_vector_int &ninput_items,
		gr_vector_const_void_star &input_items,
		gr_vector_void_star &output_items)
{
	int j=0;
	const gr_complex *in = (const gr_complex*)input_items[0];

	pthread_mutex_lock(&d_pmutex);

	// Update the FFT size from the application
	fftresize();
	windowreset();

	for(int i=0; i < noutput_items; i+=d_fftsize) 
	{
		unsigned int datasize = noutput_items - i;
		unsigned int resid = d_fftsize-d_index;

		// If we have enough input for one full FFT, do it
		if(datasize >= resid) 
		{
			const timespec currentTime = get_highres_clock();

			// Fill up residbuf with d_fftsize number of items
			memcpy(d_residbuf+d_index, &in[j], sizeof(gr_complex)*resid);
			d_index = 0;

			j += resid;
			fft(d_residbuf, d_fftsize);

			d_main_gui->UpdateWindow(true, d_fft->get_outbuf(), d_fftsize,
					NULL, 0, (float*)d_residbuf, d_fftsize,
					currentTime, true);
		}
		// Otherwise, copy what we received into the residbuf for next time
		else 
		{
			memcpy(d_residbuf+d_index, &in[j], sizeof(gr_complex)*datasize);
			d_index += datasize;
			j += datasize;
		}   
	}

	pthread_mutex_unlock(&d_pmutex);

	consume_each(j);
	return j;
}


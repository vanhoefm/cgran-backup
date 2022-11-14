/* -*- c++ -*- */
#ifndef INCLUDED_SIGPLOTGUICLASS_H
#define INCLUDED_SIGPLOTGUICLASS_H

#include <qwidget.h>
#include <qapplication.h>
#include <qlabel.h>
#include <qslider.h>
#include <spectrumUpdateEvents.h>

//#include "sigplotdisplayform.h"

#include <cmath>

#include <complex>
#include <vector>
#include <string>

class SigplotDisplayForm;

class SigplotGUIClass
{
public:
	SigplotGUIClass(const uint64_t maxDataSize, const uint64_t fftSize,
			const double newCenterFrequency,
			const double newStartFrequency, 
			const double newStopFrequency);
	~SigplotGUIClass();
	void Reset();

	void OpenSpectrumWindow(QWidget*,
			const bool frequency=true, const bool waterfall=true,
			const bool waterfall3d=true, const bool time=true,
			const bool constellation=true,
			const bool use_openGL=true);
	void SetDisplayTitle(const std::string);

	bool GetWindowOpenFlag();
	void SetWindowOpenFlag(const bool);

	void SetFrequencyRange(const double, const double, const double);
	double GetStartFrequency()const;
	double GetStopFrequency()const;
	double GetCenterFrequency()const;

	void UpdateWindow(const bool, const std::complex<float>*,
			const uint64_t, const float*,
			const uint64_t, const float*,
			const uint64_t,
			const timespec, const bool);

	float GetPowerValue()const;
	void SetPowerValue(const float);

	int GetWindowType()const;
	void SetWindowType(const int);

	int GetFFTSize()const;
	int GetFFTSizeIndex()const;
	void SetFFTSize(const int);

	timespec GetLastGUIUpdateTime()const;
	void SetLastGUIUpdateTime(const timespec);

	unsigned int GetPendingGUIUpdateEvents()const;
	void IncrementPendingGUIUpdateEvents();
	void DecrementPendingGUIUpdateEvents();
	void ResetPendingGUIUpdateEvents();

	static const long MAX_FFT_SIZE = /*1048576*/32768;
	static const long MIN_FFT_SIZE = 1024;

	QWidget* qwidget();

	void SetTimeDomainAxis(double min, double max);
	void SetConstellationAxis(double xmin, double xmax, double ymin, double ymax);
	void SetConstellationPenSize(int size);
	void SetFrequencyAxis(double min, double max);

	void SetUpdateTime(double t);

protected:

private:
	int64_t _dataPoints;
	std::string _title;
	double _centerFrequency;
	double _startFrequency;
	double _stopFrequency;
	float _powerValue;
	bool _windowOpennedFlag;
	int _windowType;
	int64_t _lastDataPointCount;
	int _fftSize;
	timespec _lastGUIUpdateTime;
	unsigned int _pendingGUIUpdateEventsCount;
	int _droppedEntriesCount;
	bool _fftBuffersCreatedFlag;
	double _updateTime;

	SigplotDisplayForm* _spectrumDisplayForm;

	std::complex<float>* _fftPoints;
	double* _realTimeDomainPoints;
	double* _imagTimeDomainPoints;
};

#endif


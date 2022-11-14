#include "sigplotguiclass.h"
#include "sigplotdisplayform.h"
#include <QEvent>
#include <QCustomEvent>

const long SigplotGUIClass::MAX_FFT_SIZE;
const long SigplotGUIClass::MIN_FFT_SIZE;

SigplotGUIClass::SigplotGUIClass(const uint64_t maxDataSize,
		const uint64_t fftSize,
		const double newCenterFrequency,
		const double newStartFrequency,
		const double newStopFrequency)
{
	_dataPoints = maxDataSize;
	if(_dataPoints < 2)
	{
		_dataPoints = 2;
	}
	_lastDataPointCount = _dataPoints;

	_fftSize = fftSize;

	_pendingGUIUpdateEventsCount = 0;
	_droppedEntriesCount = 0;

	_centerFrequency = newCenterFrequency;
	_startFrequency = newStartFrequency;
	_stopFrequency = newStopFrequency;

	_windowType = 5;

	timespec_reset(&_lastGUIUpdateTime);

	_windowOpennedFlag = false;
	_fftBuffersCreatedFlag = false;

	_powerValue = 1;
}

SigplotGUIClass::~SigplotGUIClass()
{
	/*
	if(GetWindowOpenFlag())
	{
		delete _spectrumDisplayForm;
	}
	*/

	if(_fftBuffersCreatedFlag)
	{
		delete[] _fftPoints;
		delete[] _realTimeDomainPoints;
		delete[] _imagTimeDomainPoints;
	}
}

void SigplotGUIClass::OpenSpectrumWindow(QWidget* parent,
		const bool frequency, const bool waterfall,
		const bool waterfall3d, const bool time,
		const bool constellation,
		const bool use_openGL)
{
	if(!_windowOpennedFlag)
	{
		if(!_fftBuffersCreatedFlag)
		{
			_fftPoints = new std::complex<float>[_dataPoints];
			_realTimeDomainPoints = new double[_dataPoints];
			_imagTimeDomainPoints = new double[_dataPoints];
			_fftBuffersCreatedFlag = true;


			memset(_fftPoints, 0x0, _dataPoints*sizeof(std::complex<float>));
			memset(_realTimeDomainPoints, 0x0, _dataPoints*sizeof(double));
			memset(_imagTimeDomainPoints, 0x0, _dataPoints*sizeof(double));
		}

		// Called from the Event Thread
		_spectrumDisplayForm = new SigplotDisplayForm(use_openGL, parent);

		// Toggle Windows on/off
		_spectrumDisplayForm->ToggleTabFrequency(frequency);
		_spectrumDisplayForm->ToggleTabWaterfall(waterfall);
		_spectrumDisplayForm->ToggleTabWaterfall3D(waterfall3d);
		_spectrumDisplayForm->ToggleTabTime(time);
		_spectrumDisplayForm->ToggleTabConstellation(constellation);

		_windowOpennedFlag = true;

		_spectrumDisplayForm->setSystem(this, _dataPoints, _fftSize);

		qApp->processEvents();
	}

	SetDisplayTitle(_title);
	Reset();

	qApp->postEvent(_spectrumDisplayForm,
	new QEvent(QEvent::Type(QEvent::User+3)));

	qApp->processEvents();

	timespec_reset(&_lastGUIUpdateTime);

	// Draw Blank Display
	UpdateWindow(false, NULL, 0, NULL, 0, NULL, 0, get_highres_clock(), true);

	// Set up the initial frequency axis settings
	SetFrequencyRange(_centerFrequency, _startFrequency, _stopFrequency);

	// GUI Thread only
	qApp->processEvents();
}

void SigplotGUIClass::Reset()
{
	if(GetWindowOpenFlag()) 
	{
		qApp->postEvent(_spectrumDisplayForm,
				new SpectrumFrequencyRangeEvent(_centerFrequency, 
				_startFrequency, _stopFrequency));
		qApp->postEvent(_spectrumDisplayForm, new SpectrumWindowResetEvent());
	}
	_droppedEntriesCount = 0;
}

void SigplotGUIClass::SetDisplayTitle(const std::string newString)
{
	_title.assign(newString);

	if(GetWindowOpenFlag())
	{
		qApp->postEvent(_spectrumDisplayForm,
		new SpectrumWindowCaptionEvent(_title.c_str()));
	}
}

bool SigplotGUIClass::GetWindowOpenFlag()
{
	bool returnFlag = false;
	returnFlag =  _windowOpennedFlag;
	return returnFlag;
}


void SigplotGUIClass::SetWindowOpenFlag(const bool newFlag)
{
	_windowOpennedFlag = newFlag;
}

void SigplotGUIClass::SetFrequencyRange(const double centerFreq,
		const double startFreq,
		const double stopFreq)
{
	_centerFrequency = centerFreq;
	_startFrequency = startFreq;
	_stopFrequency = stopFreq;

	_spectrumDisplayForm->SetFrequencyRange(_centerFrequency,
			_startFrequency, _stopFrequency);
}

double SigplotGUIClass::GetStartFrequency() const 
{
	double returnValue = 0.0;
	returnValue =  _startFrequency;
	return returnValue;
}

double SigplotGUIClass::GetStopFrequency() const
{
	double returnValue = 0.0;
	returnValue =  _stopFrequency;
	return returnValue;
}

double SigplotGUIClass::GetCenterFrequency() const
{
	double returnValue = 0.0;
	returnValue =  _centerFrequency;
	return returnValue;
}


void SigplotGUIClass::UpdateWindow(const bool updateDisplayFlag,
		const std::complex<float>* fftBuffer,
		const uint64_t inputBufferSize,
		const float* realTimeDomainData,
		const uint64_t realTimeDomainDataSize,
		const float* complexTimeDomainData,
		const uint64_t complexTimeDomainDataSize,
		const timespec timestamp,
		const bool lastOfMultipleFFTUpdateFlag)
{
	int64_t bufferSize = inputBufferSize;
	bool repeatDataFlag = false;
	if(bufferSize > _dataPoints)
	{
		bufferSize = _dataPoints;
	}
	int64_t timeDomainBufferSize = 0;

	if(updateDisplayFlag)
	{
		if((fftBuffer != NULL) && (bufferSize > 0))
		{
			memcpy(_fftPoints, fftBuffer, bufferSize * sizeof(std::complex<float>));
		}

		// Can't do a memcpy since ths is going from float to double data type
		if((realTimeDomainData != NULL) && (realTimeDomainDataSize > 0))
		{
			const float* realTimeDomainDataPtr = realTimeDomainData;

			double* realTimeDomainPointsPtr = _realTimeDomainPoints;
			timeDomainBufferSize = realTimeDomainDataSize;

			memset( _imagTimeDomainPoints, 0x0, realTimeDomainDataSize*sizeof(double));
			for( uint64_t number = 0; number < realTimeDomainDataSize; number++)
			{
				*realTimeDomainPointsPtr++ = *realTimeDomainDataPtr++;
			}
		}

		// Can't do a memcpy since ths is going from float to double data type
		if((complexTimeDomainData != NULL) && (complexTimeDomainDataSize > 0))
		{
			const float* complexTimeDomainDataPtr = complexTimeDomainData;

			double* realTimeDomainPointsPtr = _realTimeDomainPoints;
			double* imagTimeDomainPointsPtr = _imagTimeDomainPoints;

			timeDomainBufferSize = complexTimeDomainDataSize;
			for( uint64_t number = 0; number < complexTimeDomainDataSize; number++)
			{
				*realTimeDomainPointsPtr++ = *complexTimeDomainDataPtr++;
				*imagTimeDomainPointsPtr++ = *complexTimeDomainDataPtr++;
			}
		}
	}

	// If bufferSize is zero, then just update the display by sending over the old data
	if(bufferSize < 1)
	{
		bufferSize = _lastDataPointCount;
		repeatDataFlag = true;
	}
	else
	{
		// Since there is data this time, update the count
		_lastDataPointCount = bufferSize;
	}

	const timespec currentTime = get_highres_clock();
	const timespec lastUpdateGUITime = GetLastGUIUpdateTime();

	if((diff_timespec(currentTime, lastUpdateGUITime) > (4*_updateTime)) &&
		(GetPendingGUIUpdateEvents() > 0) && !timespec_empty(&lastUpdateGUITime)) 
	{
		// Do not update the display if too much data is pending to be displayed
		_droppedEntriesCount++;
	}
	else
	{
		// Draw the Data
		IncrementPendingGUIUpdateEvents();
		qApp->postEvent(_spectrumDisplayForm,
				new SpectrumUpdateEvent(_fftPoints, bufferSize, 
					_realTimeDomainPoints, _imagTimeDomainPoints, 
					timeDomainBufferSize, timestamp, repeatDataFlag,
					lastOfMultipleFFTUpdateFlag, currentTime, 
					_droppedEntriesCount));

		// Only reset the dropped entries counter if this is not
		// repeat data since repeat data is dropped by the display systems
		if(!repeatDataFlag)
		{
			_droppedEntriesCount = 0;
		}
	}
}

float SigplotGUIClass::GetPowerValue() const
{
	float returnValue = 0;
	returnValue = _powerValue;
	return returnValue;
}

void SigplotGUIClass::SetPowerValue(const float value)
{
	_powerValue = value;
}

int SigplotGUIClass::GetWindowType() const
{
	int returnValue = 0;
	returnValue = _windowType;
	return returnValue;
}

void SigplotGUIClass::SetWindowType(const int newType)
{
	_windowType = newType;
}

int SigplotGUIClass::GetFFTSize() const
{
	int returnValue = 0;
	returnValue = _fftSize;
	return returnValue;
}

int SigplotGUIClass::GetFFTSizeIndex() const
{
	int fftsize = GetFFTSize();
	switch(fftsize) 
	{
	case(1024): return 0; break;
	case(2048): return 1; break;
	case(4096): return 2; break;
	case(8192): return 3; break;
	case(16384): return 3; break;
	case(32768): return 3; break;
	default: return 0;
	}
}

void SigplotGUIClass::SetFFTSize(const int newSize)
{
	_fftSize = newSize;
}

timespec SigplotGUIClass::GetLastGUIUpdateTime() const
{
	timespec returnValue;
	returnValue = _lastGUIUpdateTime;
	return returnValue;
}

void SigplotGUIClass::SetLastGUIUpdateTime(const timespec newTime)
{
	_lastGUIUpdateTime = newTime;
}

unsigned int SigplotGUIClass::GetPendingGUIUpdateEvents() const
{
	unsigned int returnValue = 0;
	returnValue = _pendingGUIUpdateEventsCount;
	return returnValue;
}

void SigplotGUIClass::IncrementPendingGUIUpdateEvents()
{
	_pendingGUIUpdateEventsCount++;
}

void SigplotGUIClass::DecrementPendingGUIUpdateEvents()
{
	if(_pendingGUIUpdateEventsCount > 0)
	{
		_pendingGUIUpdateEventsCount--;
	}
}

void SigplotGUIClass::ResetPendingGUIUpdateEvents()
{
	_pendingGUIUpdateEventsCount = 0;
}


QWidget* SigplotGUIClass::qwidget()
{
	return (QWidget*)_spectrumDisplayForm;
}

void SigplotGUIClass::SetTimeDomainAxis(double min, double max)
{
	_spectrumDisplayForm->SetTimeDomainAxis(min, max);
}

void SigplotGUIClass::SetConstellationAxis(double xmin, double xmax,
		double ymin, double ymax)
{
	_spectrumDisplayForm->SetConstellationAxis(xmin, xmax, ymin, ymax);
}

void SigplotGUIClass::SetConstellationPenSize(int size)
{
	_spectrumDisplayForm->SetConstellationPenSize(size);
}


void SigplotGUIClass::SetFrequencyAxis(double min, double max)
{
	_spectrumDisplayForm->SetFrequencyAxis(min, max);
}

void SigplotGUIClass::SetUpdateTime(double t)
{
	_updateTime = t;
	_spectrumDisplayForm->SetUpdateTime(_updateTime);
}


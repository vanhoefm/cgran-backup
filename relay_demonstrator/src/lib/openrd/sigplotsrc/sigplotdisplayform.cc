#include <cmath>
#include <QColorDialog>
#include <QMessageBox>
#include "sigplotdisplayform.h"
#include "sigplotguiclass.h"

int SigplotDisplayForm::_openGLWaterfall3DFlag = -1;

SigplotDisplayForm::SigplotDisplayForm(bool useOpenGL, QWidget* parent) :
	QWidget(parent)
{
	setupUi(this);

	_useOpenGL = useOpenGL;
	_systemSpecifiedFlag = false;
	_intValidator = new QIntValidator(this);
	_intValidator->setBottom(0);
	_frequencyDisplayPlot = new FrequencyDisplayPlot(FrequencyPlotDisplayFrame);
	_waterfallDisplayPlot = new WaterfallDisplayPlot(WaterfallPlotDisplayFrame);
	_timeDomainDisplayPlot = new TimeDomainDisplayPlot(TimeDomainDisplayFrame);
	_constellationDisplayPlot = new ConstellationDisplayPlot(ConstellationDisplayFrame);
	_numRealDataPoints = 1024;
	_realFFTDataPoints = new double[_numRealDataPoints];
	_averagedValues = new double[_numRealDataPoints];
	_historyVector = new std::vector<double*>;

	AvgLineEdit->setRange(0, 500);
	MinHoldCheckBox_toggled( false );
	MaxHoldCheckBox_toggled( false );

	WaterfallMaximumIntensityWheel->setRange(-200, 0);
	WaterfallMaximumIntensityWheel->setTickCnt(50);
	WaterfallMinimumIntensityWheel->setRange(-200, 0);
	WaterfallMinimumIntensityWheel->setTickCnt(50);
	WaterfallMinimumIntensityWheel->setValue(-200);

	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		Waterfall3DMaximumIntensityWheel->setRange(-200, 0);
		Waterfall3DMaximumIntensityWheel->setTickCnt(50);
		Waterfall3DMinimumIntensityWheel->setRange(-200, 0);
		Waterfall3DMinimumIntensityWheel->setTickCnt(50);
		Waterfall3DMinimumIntensityWheel->setValue(-200);
	}

	_peakFrequency = 0;
	_peakAmplitude = -HUGE_VAL;

	_noiseFloorAmplitude = -HUGE_VAL;

	connect(_waterfallDisplayPlot, SIGNAL(UpdatedLowerIntensityLevel(const double)), 
			_frequencyDisplayPlot, SLOT(SetLowerIntensityLevel(const double)));
	connect(_waterfallDisplayPlot, SIGNAL(UpdatedUpperIntensityLevel(const double)), 
			_frequencyDisplayPlot, SLOT(SetUpperIntensityLevel(const double)));

	_frequencyDisplayPlot->SetLowerIntensityLevel(-200);
	_frequencyDisplayPlot->SetUpperIntensityLevel(-200);

	// Load up the acceptable FFT sizes...
	FFTSizeComboBox->clear();
	for(long fftSize = SigplotGUIClass::MIN_FFT_SIZE; 
			fftSize <= SigplotGUIClass::MAX_FFT_SIZE; fftSize *= 2)
	{
		FFTSizeComboBox->insertItem(FFTSizeComboBox->count(), QString("%1").arg(fftSize));
	}
	Reset();

	ToggleTabFrequency(false);
	ToggleTabWaterfall(false);
	ToggleTabWaterfall3D(false);
	ToggleTabTime(false);
	ToggleTabConstellation(false);

	// Create a timer to update plots at the specified rate
	displayTimer = new QTimer(this);
	connect(displayTimer, SIGNAL(timeout()), this, SLOT(UpdateGuiTimer()));
}

SigplotDisplayForm::~SigplotDisplayForm()
{
	delete _intValidator;

	delete[] _realFFTDataPoints;
	delete[] _averagedValues;

	for(unsigned int count = 0; count < _historyVector->size(); count++)
	{
		delete[] _historyVector->operator[](count);
	}

	delete _historyVector;

	displayTimer->stop();
	delete displayTimer;
}

void SigplotDisplayForm::setSystem( SigplotGUIClass * newSystem, 
		const uint64_t numFFTDataPoints, 
		const uint64_t numTimeDomainDataPoints )
{
	ResizeBuffers(numFFTDataPoints, numTimeDomainDataPoints);

	if(newSystem != NULL)
	{
		_system = newSystem;
		_systemSpecifiedFlag = true;
	}
	else
	{
		_systemSpecifiedFlag = false;
	}
}

void SigplotDisplayForm::newFrequencyData( const SpectrumUpdateEvent* spectrumUpdateEvent)
{
	const std::complex<float>* complexDataPoints = spectrumUpdateEvent->getFFTPoints();
	const uint64_t numFFTDataPoints = spectrumUpdateEvent->getNumFFTDataPoints();
	const double* realTimeDomainDataPoints = spectrumUpdateEvent->getRealTimeDomainPoints();
	const double* imagTimeDomainDataPoints = spectrumUpdateEvent->getImagTimeDomainPoints();
	const uint64_t numTimeDomainDataPoints = spectrumUpdateEvent->getNumTimeDomainDataPoints();
	const timespec dataTimestamp = spectrumUpdateEvent->getDataTimestamp();
	const bool repeatDataFlag = spectrumUpdateEvent->getRepeatDataFlag();
	const bool lastOfMultipleUpdatesFlag = spectrumUpdateEvent->getLastOfMultipleUpdateFlag();
	const timespec generatedTimestamp = spectrumUpdateEvent->getEventGeneratedTimestamp();

	// REMEMBER: The dataTimestamp is NOT valid when the repeat data flag is true...
	ResizeBuffers(numFFTDataPoints, numTimeDomainDataPoints);

	// Calculate the Magnitude of the complex point
	const std::complex<float>* complexDataPointsPtr = complexDataPoints+numFFTDataPoints/2;
	double* realFFTDataPointsPtr = _realFFTDataPoints;

	double sumMean, localPeakAmplitude, localPeakFrequency;
	const double fftBinSize = (_stopFrequency-_startFrequency) /
	static_cast<double>(numFFTDataPoints);
	localPeakAmplitude = -HUGE_VAL;
	sumMean = 0.0;

	// Run this twice to perform the fftshift operation on the data here as well
	std::complex<float> scaleFactor = std::complex<float>((float)numFFTDataPoints);
	for(uint64_t point = 0; point < numFFTDataPoints/2; point++)
	{
		std::complex<float> pt = (*complexDataPointsPtr) / scaleFactor;
		*realFFTDataPointsPtr = 10.0*log10((pt.real() * pt.real() + pt.imag()*pt.imag()) + 1e-20);

		if(*realFFTDataPointsPtr > localPeakAmplitude) 
		{
			localPeakFrequency = static_cast<float>(point) * fftBinSize;
			localPeakAmplitude = *realFFTDataPointsPtr;
		}
		sumMean += *realFFTDataPointsPtr;

		complexDataPointsPtr++;
		realFFTDataPointsPtr++;
	}

	// This loop takes the first half of the input data and puts it in the 
	// second half of the plotted data
	complexDataPointsPtr = complexDataPoints;
	for(uint64_t point = 0; point < numFFTDataPoints/2; point++)
	{
		std::complex<float> pt = (*complexDataPointsPtr) / scaleFactor;
		*realFFTDataPointsPtr = 10.0*log10((pt.real() * pt.real() + pt.imag()*pt.imag()) + 1e-20);

		if(*realFFTDataPointsPtr > localPeakAmplitude) 
		{
			localPeakFrequency = static_cast<float>(point) * fftBinSize;
			localPeakAmplitude = *realFFTDataPointsPtr;
		}
		sumMean += *realFFTDataPointsPtr;

		complexDataPointsPtr++;
		realFFTDataPointsPtr++;
	}

	// don't update the averaging history if this is repeated data
	if(!repeatDataFlag)
	{
		_AverageHistory(_realFFTDataPoints);

		// Only use the local info if we are not repeating data
		_peakAmplitude = localPeakAmplitude;
		_peakFrequency = localPeakFrequency;

		// calculate the spectral mean
		// +20 because for the comparison below we only want to throw out bins
		// that are significantly higher (and would, thus, affect the mean more)
		const double meanAmplitude = (sumMean / numFFTDataPoints) + 20.0;

		// now throw out any bins higher than the mean
		sumMean = 0.0;
		uint64_t newNumDataPoints = numFFTDataPoints;
		for(uint64_t number = 0; number < numFFTDataPoints; number++)
		{
			if (_realFFTDataPoints[number] <= meanAmplitude)
				sumMean += _realFFTDataPoints[number];
			else
				newNumDataPoints--;
		}

		if (newNumDataPoints == 0)             // in the odd case that all
			_noiseFloorAmplitude = meanAmplitude; // amplitudes are equal!
		else
			_noiseFloorAmplitude = sumMean / newNumDataPoints;
	}

	if(lastOfMultipleUpdatesFlag)
	{
		int tabindex = SpectrumTypeTab->currentIndex();
		if(tabindex == d_plot_fft) 
		{
			_frequencyDisplayPlot->PlotNewData(_averagedValues, numFFTDataPoints, 
					 _noiseFloorAmplitude, _peakFrequency, 
					 _peakAmplitude, d_update_time);
		}
		if(tabindex == d_plot_time) 
		{
			_timeDomainDisplayPlot->PlotNewData(realTimeDomainDataPoints, 
					imagTimeDomainDataPoints, 
					numTimeDomainDataPoints,
					d_update_time);
		}
		if(tabindex == d_plot_constellation) 
		{
			_constellationDisplayPlot->PlotNewData(realTimeDomainDataPoints, 
					imagTimeDomainDataPoints, 
					numTimeDomainDataPoints,
					d_update_time);
		}

		// Don't update the repeated data for the waterfall
		if(!repeatDataFlag)
		{
			if(tabindex == d_plot_waterfall) 
			{
				_waterfallDisplayPlot->PlotNewData(_realFFTDataPoints, numFFTDataPoints, 
						d_update_time, dataTimestamp, 
						spectrumUpdateEvent->getDroppedFFTFrames());
			}
		}


		// Tell the system the GUI has been updated
		if(_systemSpecifiedFlag)
		{
			_system->SetLastGUIUpdateTime(generatedTimestamp);
			_system->DecrementPendingGUIUpdateEvents();
		}
	}
}

void SigplotDisplayForm::resizeEvent( QResizeEvent *e )
{
	QSize s;
	s.setWidth(FrequencyPlotDisplayFrame->width());
	s.setHeight(FrequencyPlotDisplayFrame->height());
	//emit 
	_frequencyDisplayPlot->resizeSlot(&s);

	s.setWidth(TimeDomainDisplayFrame->width());
	s.setHeight(TimeDomainDisplayFrame->height());
	//emit 
	_timeDomainDisplayPlot->resizeSlot(&s);

	s.setWidth(WaterfallPlotDisplayFrame->width());
	s.setHeight(WaterfallPlotDisplayFrame->height());
	//emit
	_waterfallDisplayPlot->resizeSlot(&s);

	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		s.setWidth(Waterfall3DPlotDisplayFrame->width());
		s.setHeight(Waterfall3DPlotDisplayFrame->height());
		//emit _waterfall3DDisplayPlot->resizeSlot(&s);
	}

	s.setWidth(ConstellationDisplayFrame->width());
	s.setHeight(ConstellationDisplayFrame->height());
	//emit
	_constellationDisplayPlot->resizeSlot(&s);
}

void SigplotDisplayForm::customEvent( QEvent * e)
{
	if(e->type() == QEvent::User+3)
	{
		if(_systemSpecifiedFlag)
		{
			WindowComboBox->setCurrentIndex(_system->GetWindowType());
			FFTSizeComboBox->setCurrentIndex(_system->GetFFTSizeIndex());
		}

		waterfallMinimumIntensityChangedCB(WaterfallMinimumIntensityWheel->value());
		waterfallMaximumIntensityChangedCB(WaterfallMaximumIntensityWheel->value());

		// If the video card doesn't support OpenGL then don't display the 3D Waterfall
		if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
		{
			waterfall3DMinimumIntensityChangedCB(Waterfall3DMinimumIntensityWheel->value());
			waterfall3DMaximumIntensityChangedCB(Waterfall3DMaximumIntensityWheel->value());
		}

		if(_openGLWaterfall3DFlag != 1)
		{
			ToggleTabWaterfall3D(false);
		}

		// Clear any previous display
		Reset();
	}
	else if(e->type() == 10005)
	{
		SpectrumUpdateEvent* spectrumUpdateEvent = (SpectrumUpdateEvent*)e;
		newFrequencyData(spectrumUpdateEvent);
	}
	else if(e->type() == 10008)
	{
		setWindowTitle(((SpectrumWindowCaptionEvent*)e)->getLabel());
	}
	else if(e->type() == 10009)
	{
		Reset();
		if(_systemSpecifiedFlag)
		{
			_system->ResetPendingGUIUpdateEvents();
		}
	}
	else if(e->type() == 10010)
	{
		_startFrequency = ((SpectrumFrequencyRangeEvent*)e)->GetStartFrequency();
		_stopFrequency = ((SpectrumFrequencyRangeEvent*)e)->GetStopFrequency();
		_centerFrequency  = ((SpectrumFrequencyRangeEvent*)e)->GetCenterFrequency();

		UseRFFrequenciesCB(UseRFFrequenciesCheckBox->isChecked());
	}
}

void SigplotDisplayForm::UpdateGuiTimer()
{
	// This is called by the displayTimer and redraws the canvases of
	// all of the plots.
	_frequencyDisplayPlot->canvas()->update();
	_waterfallDisplayPlot->canvas()->update();
	_timeDomainDisplayPlot->canvas()->update();
	_constellationDisplayPlot->canvas()->update();
}


void SigplotDisplayForm::AvgLineEdit_valueChanged( int value )
{
	SetAverageCount(value);
}


void SigplotDisplayForm::MaxHoldCheckBox_toggled( bool newState )
{
	MaxHoldResetBtn->setEnabled(newState);
	_frequencyDisplayPlot->SetMaxFFTVisible(newState);
	MaxHoldResetBtn_clicked();
}


void SigplotDisplayForm::MinHoldCheckBox_toggled( bool newState )
{
	MinHoldResetBtn->setEnabled(newState);
	_frequencyDisplayPlot->SetMinFFTVisible(newState);
	MinHoldResetBtn_clicked();
}


void SigplotDisplayForm::MinHoldResetBtn_clicked()
{
	_frequencyDisplayPlot->ClearMinData();
	_frequencyDisplayPlot->replot();
}


void SigplotDisplayForm::MaxHoldResetBtn_clicked()
{
	_frequencyDisplayPlot->ClearMaxData();
	_frequencyDisplayPlot->replot();
}


void SigplotDisplayForm::TabChanged(int index)
{
	// This might be dangerous to call this with NULL
	resizeEvent(NULL);  
}

void SigplotDisplayForm::SetFrequencyRange(const double newCenterFrequency,
		const double newStartFrequency, const double newStopFrequency)
{
	double fdiff;
	if(UseRFFrequenciesCheckBox->isChecked()) 
	{
		fdiff = newCenterFrequency;
	}
	else
	{
		fdiff = std::max(fabs(newStartFrequency), fabs(newStopFrequency));
	}

	if(fdiff > 0) 
	{
		std::string strunits[4] = {"Hz", "kHz", "MHz", "GHz"};
		std::string strtime[4] = {"sec", "ms", "us", "ns"};
		double units10 = floor(log10(fdiff));
		double units3  = std::max(floor(units10 / 3.0), 0.0);
		double units = pow(10, (units10-fmod(units10, 3.0)));
		int iunit = static_cast<int>(units3);

		_startFrequency = newStartFrequency;
		_stopFrequency = newStopFrequency;
		_centerFrequency = newCenterFrequency;

		_frequencyDisplayPlot->SetFrequencyRange(_startFrequency,
				_stopFrequency,
				_centerFrequency,
				UseRFFrequenciesCheckBox->isChecked(),
				units, strunits[iunit]);
		_waterfallDisplayPlot->SetFrequencyRange(_startFrequency,
				_stopFrequency,
				_centerFrequency,
				UseRFFrequenciesCheckBox->isChecked(),
				units, strunits[iunit]);
		_timeDomainDisplayPlot->SetSampleRate(_stopFrequency - _startFrequency,
				units, strtime[iunit]);
	}
}

int SigplotDisplayForm::GetAverageCount()
{
	return _historyVector->size();
}

void SigplotDisplayForm::SetAverageCount(const int newCount)
{
	if(newCount > -1)
	{
		if(newCount != static_cast<int>(_historyVector->size()))
		{
			std::vector<double*>::iterator pos;
			while(newCount < static_cast<int>(_historyVector->size()))
			{
				pos = _historyVector->begin();
				delete[] (*pos);
				_historyVector->erase(pos);
			}

			while(newCount > static_cast<int>(_historyVector->size()))
			{
				_historyVector->push_back(new double[_numRealDataPoints]);
			}
			AverageDataReset();
		}
	}
}

void SigplotDisplayForm::_AverageHistory(const double* newBuffer)
{
	if(_numRealDataPoints > 0)
	{
		if(_historyVector->size() > 0)
		{
			memcpy(_historyVector->operator[](_historyEntry), newBuffer,
					_numRealDataPoints*sizeof(double));

			// Increment the next location to store data
			_historyEntryCount++;
			if(_historyEntryCount > static_cast<int>(_historyVector->size()))
			{
				_historyEntryCount = _historyVector->size();
			}
			_historyEntry = (++_historyEntry)%_historyVector->size();

			// Total up and then average the values
			double sum;
			for(uint64_t location = 0; location < _numRealDataPoints; location++)
			{
				sum = 0;
				for(int number = 0; number < _historyEntryCount; number++)
				{
					sum += _historyVector->operator[](number)[location];
				}
				_averagedValues[location] = sum/static_cast<double>(_historyEntryCount);
			}
		}
		else
		{
			memcpy(_averagedValues, newBuffer, _numRealDataPoints*sizeof(double));
		}
	}
}

void SigplotDisplayForm::ResizeBuffers( const uint64_t numFFTDataPoints,
				    const uint64_t /*numTimeDomainDataPoints*/ )
{
	// Convert from Complex to Real for certain Displays
	if(_numRealDataPoints != numFFTDataPoints)
	{
		_numRealDataPoints = numFFTDataPoints;
		delete[] _realFFTDataPoints;
		delete[] _averagedValues;

		_realFFTDataPoints = new double[_numRealDataPoints];
		_averagedValues = new double[_numRealDataPoints];
		memset(_realFFTDataPoints, 0x0, _numRealDataPoints*sizeof(double));

		const int historySize = _historyVector->size();
		SetAverageCount(0); // Clear the existing history
		SetAverageCount(historySize);

		Reset();
	}
}

void SigplotDisplayForm::Reset()
{
	AverageDataReset();

	_waterfallDisplayPlot->Reset();
}


void SigplotDisplayForm::AverageDataReset()
{
	_historyEntry = 0;
	_historyEntryCount = 0;

	memset(_averagedValues, 0x0, _numRealDataPoints*sizeof(double));

	MaxHoldResetBtn_clicked();
	MinHoldResetBtn_clicked();
}


void SigplotDisplayForm::closeEvent( QCloseEvent *e )
{
	if(_systemSpecifiedFlag)
	{
		_system->SetWindowOpenFlag(false);
	}

	qApp->processEvents();

	QWidget::closeEvent(e);
}


void SigplotDisplayForm::WindowTypeChanged( int newItem )
{
	if(_systemSpecifiedFlag)
	{
		_system->SetWindowType(newItem);
	}
}

void SigplotDisplayForm::UseRFFrequenciesCB( bool useRFFlag )
{
	SetFrequencyRange(_centerFrequency, _startFrequency, _stopFrequency);
}


void SigplotDisplayForm::waterfallMaximumIntensityChangedCB( double newValue )
{
	if(newValue > WaterfallMinimumIntensityWheel->value())
	{
		WaterfallMaximumIntensityLabel->setText(QString("%1 dB").arg(newValue, 0, 'f', 0));
	}
	else
	{
		WaterfallMaximumIntensityWheel->setValue(WaterfallMinimumIntensityWheel->value());
	}

	_waterfallDisplayPlot->SetIntensityRange(WaterfallMinimumIntensityWheel->value(),
	WaterfallMaximumIntensityWheel->value());
}


void SigplotDisplayForm::waterfallMinimumIntensityChangedCB( double newValue )
{
	if(newValue < WaterfallMaximumIntensityWheel->value())
	{
		WaterfallMinimumIntensityLabel->setText(QString("%1 dB").arg(newValue, 0, 'f', 0));
	}
	else
	{
		WaterfallMinimumIntensityWheel->setValue(WaterfallMaximumIntensityWheel->value());
	}

	_waterfallDisplayPlot->SetIntensityRange(WaterfallMinimumIntensityWheel->value(),
	WaterfallMaximumIntensityWheel->value());
}

void SigplotDisplayForm::waterfall3DMaximumIntensityChangedCB( double newValue )
{
	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		if(newValue > Waterfall3DMinimumIntensityWheel->value())
		{
			Waterfall3DMaximumIntensityLabel->setText(QString("%1 dB").arg(newValue, 0, 'f', 0));
		}
		else
		{
			Waterfall3DMaximumIntensityWheel->setValue(Waterfall3DMinimumIntensityWheel->value());
		}
	}
}


void SigplotDisplayForm::waterfall3DMinimumIntensityChangedCB( double newValue )
{
	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		if(newValue < Waterfall3DMaximumIntensityWheel->value())
		{
			Waterfall3DMinimumIntensityLabel->setText(QString("%1 dB").arg(newValue, 0, 'f', 0));
		}
		else
		{
			Waterfall3DMinimumIntensityWheel->setValue(Waterfall3DMaximumIntensityWheel->value());
		}
	}
}


void SigplotDisplayForm::FFTComboBoxSelectedCB( const QString &fftSizeString )
{
	if(_systemSpecifiedFlag)
	{
		_system->SetFFTSize(fftSizeString.toLong());
	}
}


void SigplotDisplayForm::WaterfallAutoScaleBtnCB()
{
	double minimumIntensity = _noiseFloorAmplitude - 5;
	if(minimumIntensity < WaterfallMinimumIntensityWheel->minValue())
	{
		minimumIntensity = WaterfallMinimumIntensityWheel->minValue();
	}
	WaterfallMinimumIntensityWheel->setValue(minimumIntensity);
	double maximumIntensity = _peakAmplitude + 10;
	if(maximumIntensity > WaterfallMaximumIntensityWheel->maxValue())
	{
		maximumIntensity = WaterfallMaximumIntensityWheel->maxValue();
	}
	WaterfallMaximumIntensityWheel->setValue(maximumIntensity);
	waterfallMaximumIntensityChangedCB(maximumIntensity);
}

void SigplotDisplayForm::Waterfall3DAutoScaleBtnCB()
{
	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		double minimumIntensity = _noiseFloorAmplitude - 5;
		if(minimumIntensity < Waterfall3DMinimumIntensityWheel->minValue())
		{
			minimumIntensity = Waterfall3DMinimumIntensityWheel->minValue();
		}
		Waterfall3DMinimumIntensityWheel->setValue(minimumIntensity);
		double maximumIntensity = _peakAmplitude + 10;
		if(maximumIntensity > Waterfall3DMaximumIntensityWheel->maxValue())
		{
			maximumIntensity = Waterfall3DMaximumIntensityWheel->maxValue();
		}
		Waterfall3DMaximumIntensityWheel->setValue(maximumIntensity);
		waterfallMaximumIntensityChangedCB(maximumIntensity);
	}
}

void SigplotDisplayForm::WaterfallIntensityColorTypeChanged( int newType )
{
	QColor lowIntensityColor;
	QColor highIntensityColor;
	if(newType == WaterfallDisplayPlot::INTENSITY_COLOR_MAP_TYPE_USER_DEFINED)
	{
		// Select the Low Intensity Color
		lowIntensityColor = _waterfallDisplayPlot->GetUserDefinedLowIntensityColor();
		if(!lowIntensityColor.isValid())
		{
			lowIntensityColor = Qt::black;
		}
		QMessageBox::information(this, "Low Intensity Color Selection", "In the next window, select the low intensity color for the waterfall display",  QMessageBox::Ok);
		lowIntensityColor = QColorDialog::getColor(lowIntensityColor, this);

		// Select the High Intensity Color
		highIntensityColor = _waterfallDisplayPlot->GetUserDefinedHighIntensityColor();
		if(!highIntensityColor.isValid())
		{
			highIntensityColor = Qt::white;
		}
		QMessageBox::information(this, "High Intensity Color Selection", "In the next window, select the high intensity color for the waterfall display",  QMessageBox::Ok);
		highIntensityColor = QColorDialog::getColor(highIntensityColor, this);
	}

	_waterfallDisplayPlot->SetIntensityColorMapType(newType, lowIntensityColor, highIntensityColor);
}

void SigplotDisplayForm::Waterfall3DIntensityColorTypeChanged( int newType )
{
	if((QGLFormat::hasOpenGL()) && (_useOpenGL)) 
	{
		QColor lowIntensityColor;
		QColor highIntensityColor;
		if(newType == Waterfall3DDisplayPlot::INTENSITY_COLOR_MAP_TYPE_USER_DEFINED)
		{
			// Select the Low Intensity Color
			lowIntensityColor = _waterfallDisplayPlot->GetUserDefinedLowIntensityColor();
			if(!lowIntensityColor.isValid())
			{
				lowIntensityColor = Qt::black;
			}
			QMessageBox::information(this, "Low Intensity Color Selection", "In the next window, select the low intensity color for the waterfall display",  QMessageBox::Ok);
			lowIntensityColor = QColorDialog::getColor(lowIntensityColor, this);

			// Select the High Intensity Color
			highIntensityColor = _waterfallDisplayPlot->GetUserDefinedHighIntensityColor();
			if(!highIntensityColor.isValid())
			{
				highIntensityColor = Qt::white;
			}
			QMessageBox::information(this, "High Intensity Color Selection", "In the next window, select the high intensity color for the waterfall display",  QMessageBox::Ok);
			highIntensityColor = QColorDialog::getColor(highIntensityColor, this);
		}
	}
}


void SigplotDisplayForm::ToggleTabFrequency(const bool state)
{
	if(state == true) 
	{
		if(d_plot_fft == -1) 
		{
			SpectrumTypeTab->addTab(FrequencyPage, "Frequency Display");
			d_plot_fft = SpectrumTypeTab->count()-1;
		}
	}
	else 
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(FrequencyPage));
		d_plot_fft = -1;
	}
}

void SigplotDisplayForm::ToggleTabWaterfall(const bool state)
{
	if(state == true) 
	{
		if(d_plot_waterfall == -1) 
		{
			SpectrumTypeTab->addTab(WaterfallPage, "Waterfall Display");
			d_plot_waterfall = SpectrumTypeTab->count()-1;
		}
	}
	else 
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(WaterfallPage));
		d_plot_waterfall = -1;
	}
}

void SigplotDisplayForm::ToggleTabWaterfall3D(const bool state)
{
	if(state == true) 
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(Waterfall3DPage));
		d_plot_waterfall3d = -1;
		fprintf(stderr, "\nWARNING: The Waterfall3D plot has been disabled until we get it working.\n\n");
	}
	else
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(Waterfall3DPage));
		d_plot_waterfall3d = -1;
	}
}

void SigplotDisplayForm::ToggleTabTime(const bool state)
{
	if(state == true) 
	{
		if(d_plot_time == -1) 
		{
			SpectrumTypeTab->addTab(TimeDomainPage, "Time Domain Display");
			d_plot_time = SpectrumTypeTab->count()-1;
		}
	}
	else 
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(TimeDomainPage));
		d_plot_time = -1;
	}
}

void SigplotDisplayForm::ToggleTabConstellation(const bool state)
{
	if(state == true) 
	{
		if(d_plot_constellation == -1) 
		{
			SpectrumTypeTab->addTab(ConstellationPage, "Constellation Display");
			d_plot_constellation = SpectrumTypeTab->count()-1;
		}
	}
	else 
	{
		SpectrumTypeTab->removeTab(SpectrumTypeTab->indexOf(ConstellationPage));
		d_plot_constellation = -1;
	}
}


void SigplotDisplayForm::SetTimeDomainAxis(double min, double max)
{
	_timeDomainDisplayPlot->set_yaxis(min, max);
}

void SigplotDisplayForm::SetConstellationAxis(double xmin, double xmax,
		double ymin, double ymax)
{
	_constellationDisplayPlot->set_axis(xmin, xmax, ymin, ymax);
}

void SigplotDisplayForm::SetConstellationPenSize(int size)
{
	_constellationDisplayPlot->set_pen_size( size );
}

void SigplotDisplayForm::SetFrequencyAxis(double min, double max)
{
	_frequencyDisplayPlot->set_yaxis(min, max);
}

void SigplotDisplayForm::SetUpdateTime(double t)
{
	d_update_time = t;
	// QTimer class takes millisecond input
	displayTimer->start(d_update_time*1000);
}


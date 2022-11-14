/* -*- c++ -*- */
#ifndef INCLUDED_SIGPLOT_H
#define INCLUDED_SIGPLOT_H

#include <qapplication.h>
#include "sigplotguiclass.h"

class sigplot_event : public QEvent
{
private:
	pthread_mutex_t *pmutex;

public:
	sigplot_event(pthread_mutex_t *mut) : 
		QEvent((QEvent::Type)(QEvent::User+101))
	{
		pmutex = mut;
	}

	void lock()
	{
		pthread_mutex_lock(pmutex);
	}

	void unlock()
	{
		pthread_mutex_unlock(pmutex);
	}
};

class sigplot_obj : public QObject
{
public:
	sigplot_obj(QObject *p) : 
		QObject(p)
	{ 
	}

	void customEvent(QEvent *e)
	{
		if(e->type() == (QEvent::Type)(QEvent::User+101)) 
		{
			sigplot_event *qt = (sigplot_event*)e;
			qt->unlock();
		}
	}
};

#endif /* INCLUDED_SIGPLOT_H */


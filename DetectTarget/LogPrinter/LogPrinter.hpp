#pragma once
#include <string>
#include "../Models/LogLevel.hpp"
#include <iostream>

class LogPrinter
{
public:
	static void PrintLogs(std::string text, LogLevel logLevel);

};

inline void LogPrinter::PrintLogs(std::string text, LogLevel logLevel)
{
	switch (logLevel)
	{
	case LogLevel::Error:
	{
		std::cout << "ERROR => ";
		break;
	}
	case LogLevel::Info:
	{
		std::cout << "INFO => ";
		break;
	}
	case LogLevel::Waring:
	{
		std::cout << "WARNING => ";
		break;
	}
	default:
	{
		std::cout << "Not Defined => ";
		break;
	};
	}

	std::cout << text << std::endl;
}

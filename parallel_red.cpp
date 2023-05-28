#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>

void min(int* arr, int n)
{
   int min_val = 10000;
#pragma omp parallel for reduction(min:min_val)
   for (int i = 0; i < n; i++)
   {
      std::cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
      if (arr[i] < min_val)
      {
         min_val = arr[i];
      }
   }
   std::cout << "\n\nmin_val = " << min_val << std::endl;
}

void max(int* arr, int n)
{
   int max_val = 0;
#pragma omp parallel for reduction(max:max_val)
   for (int i = 0; i < n; i++)
   {
      std::cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
      if (arr[i] > max_val)
      {
         max_val = arr[i];
      }
   }
   std::cout << "\n\nmax_val = " << max_val << std::endl;
}

void avg(int* arr, int n)
{
   float avg = 0, sum = 0;
#pragma omp parallel reduction(+:sum)
   {
#pragma omp for
      for (int i = 0; i < n; i++)
      {
         sum = sum + arr[i];
         std::cout << "\nthread id = " << omp_get_thread_num() << " and i = " << i;
      }
   }
   std::cout << "\n\nSum = " << sum << std::endl;
   avg = sum / n;
   std::cout << "\nAverage = " << avg << std::endl;
}

int main()
{
   omp_set_num_threads(4);
   int n;

   std::cout << "Enter the number of elements in the array: ";
   std::cin >> n;

   int* arr = new int[n];

   srand(time(0));
   for (int i = 0; i < n; ++i)
   {
      arr[i] = rand() % 100;
   }

   std::cout << "\nArray elements are: ";
   for (int i = 0; i < n; i++)
   {
      std::cout << arr[i] << ",";
   }

   min(arr, n);
   max(arr, n);
   avg(arr, n);

   delete[] arr;

   return 0;
}

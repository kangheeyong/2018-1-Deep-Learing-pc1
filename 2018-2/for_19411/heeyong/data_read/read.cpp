#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <stdlib.h>


int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("USAGE : %s [the data path]\n",argv[0]);
        return 0;
    }

    FILE *fd;

    fd = fopen(argv[1],"r");
    if (fd == NULL)
    {
        printf("file read fail\n");
        return 0;
    }
    else
    {

        int temp[1024];
        for(int i = 0 ; i < 3650 ; i++)
        {
            for(int j = 0 ; j < 3648; j++)
            {
                
                fread(temp,sizeof(int), 1 , fd);
                if((j > 3644) && (i > 3646))
                {
                    printf("%d ", temp[0]);
                }
            }
            if(i > 3646) printf("\n");
        }
      
        fclose(fd);
    }


    return 0;
}

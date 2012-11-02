! Author: Gabriel Altay, in SPHRAY

function pluecker(dir, dist, s2b, s2t) result( hit )

  real, intent(in) :: s2b(3)         !< vector from ray start to lower cell corner
  real, intent(in) :: s2t(3)         !< vector from ray start to upper cell corner
  real  :: e2b(3)       !< vector from ray end to lower cell corner
  real  :: e2t(3)       !< vector from ray end to upper cell corner

  real, intent(in) :: dist
  real, intent(in) :: dir(3)
  logical :: hit              !< true or false result
  

  integer :: class
  integer :: i

  class = 0
  do i = 1, 3
     if ( dir(i) >= 0) class = class + 2**(i-1)
  end do

  hit = .false.

  e2b = s2b - dir * dist
  e2t = s2t - dir * dist

  ! branch on ray direction
  !---------------------------
  select case( class )

     ! MMM
     !-----------
  case(0)

     if(s2b(1) > 0 .or. s2b(2) > 0 .or. s2b(3) > 0) return ! on negative part of ray 
     if(e2t(1) < 0 .or. e2t(2) < 0 .or. e2t(3) < 0) return ! past length of ray      

     if ( dir(1)*s2b(2) - dir(2)*s2t(1) < 0 .or.  &
          dir(1)*s2t(2) - dir(2)*s2b(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) > 0       ) return
     
     ! PMM
     !-----------
  case(1)
     
     if(s2t(1) < 0 .or. s2b(2) > 0 .or. s2b(3) > 0) return ! on negative part of ray 
     if(e2b(1) > 0 .or. e2t(2) < 0 .or. e2t(3) < 0) return ! past length of ray      
     
     if ( dir(1)*s2t(2) - dir(2)*s2t(1) < 0 .or.  &
          dir(1)*s2b(2) - dir(2)*s2b(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) > 0       ) return
     
     ! MPM
     !-----------
  case(2)
     
     if(s2b(1) > 0 .or. s2t(2) < 0 .or. s2b(3) > 0) return ! on negative part of ray 
     if(e2t(1) < 0 .or. e2b(2) > 0 .or. e2t(3) < 0) return ! past length of ray      
     
     if ( dir(1)*s2b(2) - dir(2)*s2b(1) < 0 .or.  &
          dir(1)*s2t(2) - dir(2)*s2t(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) > 0       ) return
     
     ! PPM
     !-----------
  case(3)
     
     if(s2t(1) < 0 .or. s2t(2) < 0 .or. s2b(3) > 0) return ! on negative part of ray 
     if(e2b(1) > 0 .or. e2b(2) > 0 .or. e2t(3) < 0) return ! past length of ray      
     
     if ( dir(1)*s2t(2) - dir(2)*s2b(1) < 0 .or.  &
          dir(1)*s2b(2) - dir(2)*s2t(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) > 0       ) return
     
     ! MMP
     !-----------
  case(4)
     
     if(s2b(1) > 0 .or. s2b(2) > 0 .or. s2t(3) < 0) return ! on negative part of ray 
     if(e2t(1) < 0 .or. e2t(2) < 0 .or. e2b(3) > 0) return ! past length of ray      
     
     if ( dir(1)*s2b(2) - dir(2)*s2t(1) < 0 .or.  &
          dir(1)*s2t(2) - dir(2)*s2b(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) > 0       ) return
     
     
     ! PMP
     !-----------
  case(5)
     
     if(s2t(1) < 0 .or. s2b(2) > 0 .or. s2t(3) < 0) return ! on negative part of ray 
     if(e2b(1) > 0 .or. e2t(2) < 0 .or. e2b(3) > 0) return ! past length of ray      
     
     if ( dir(1)*s2t(2) - dir(2)*s2t(1) < 0 .or.  &
          dir(1)*s2b(2) - dir(2)*s2b(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) > 0       ) return
     
     
     ! MPP
     !-----------
  case(6)
     
     if(s2b(1) > 0 .or. s2t(2) < 0 .or. s2t(3) < 0) return ! on negative part of ray 
     if(e2t(1) < 0 .or. e2b(2) > 0 .or. e2b(3) > 0) return ! past length of ray      
     
     if ( dir(1)*s2b(2) - dir(2)*s2b(1) < 0 .or.  &
          dir(1)*s2t(2) - dir(2)*s2t(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) > 0       ) return
     
     ! PPP
     !-----------
  case(7)
     
     if(s2t(1) < 0 .or. s2t(2) < 0 .or. s2t(3) < 0) return ! on negative part of ray 
     if(e2b(1) > 0 .or. e2b(2) > 0 .or. e2b(3) > 0) return ! past length of ray      
     
     if ( dir(1)*s2t(2) - dir(2)*s2b(1) < 0 .or.  &
          dir(1)*s2b(2) - dir(2)*s2t(1) > 0 .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) > 0 .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) < 0 .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) < 0 .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) > 0       ) return
     
  end select
  
  hit=.true.
  
end function pluecker

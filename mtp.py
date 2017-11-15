import re
import sys
from mininet.log import setLogLevel, info, error
from mininet.topo import Topo
from mininet.link import Intf
from mininet.util import quietRun

class Mtp(Topo):


    def __init__(self):
        Topo.__init__(self)

        h1 = self.addHost('h1', mac = '00:00:00:00:00:01', ip = '10.0.0.1/24')
        h2 = self.addHost('h2', mac = '00:00:00:00:00:02', ip = '10.0.0.2/24')
        #h3 = self.addHost('h3', mac = '00:00:00:00:00:03', ip = '10.0.0.3/24')
        #h4 = self.addHost('h4', mac = '00:00:00:00:00:04', ip = '10.0.0.4/24')
        s1 = self.addSwitch('s1', protocols=["OpenFlow13"] )
	s2 = self.addSwitch('s2', protocols=["OpenFlow13"] )
	s3 = self.addSwitch('s3', protocols=["OpenFlow13"] )
	s4 = self.addSwitch('s4', protocols=["OpenFlow13"] )
	s5 = self.addSwitch('s5', protocols=["OpenFlow13"] )

        #switch2 = self.addSwitch('s2', protocols=["OpenFlow13"], fail)
        #inetfname = 'vboxnet0'
        #checkIntf(inetfname)
        #_inetf = Intf(inetfname,node=switch)

        #add links
        self.addLink(h1,s1)
        self.addLink(h2,s5)
        self.addLink(s1,s2)
        self.addLink(s1,s3)
	self.addLink(s2,s4)
	self.addLink(s4,s5)
	self.addLink(s3,s5)


topos = {'mtp': (lambda:Mtp())}

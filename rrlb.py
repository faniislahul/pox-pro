"""
Author Haichen Shen
Module to perform round-robin load balancing.
"""

from pox.core import core    # the POX core object
import pox.openflow.libopenflow_01 as of
from pox.lib.revent import *    # event system
from pox.lib.util import dpidToStr
from pox.lib.packet.ethernet import ethernet    # handle ethernet
from pox.lib.packet.arp import arp    # handle arp
from pox.lib.addresses import IPAddr    # ip address
from pox.lib.addresses import EthAddr    # ethernet address
import time

log = core.getLogger()

IDLE_TIMEOUT = 60    # in seconds
HARD_TIMEOUT = 0    # infinity

LOAD_BALANCER_IP = IPAddr('10.0.0.1')    # '10.0.0.254'
LOAD_BALANCER_MAC = EthAddr('00:00:00:00:00:01')    # '00:00:00:00:00:FE'

class LoadBalancer (EventMixin):    # 

  class Server:    # 
    def __init__ (self, ip, mac, port):
      self.ip = IPAddr(ip)    # set the ip address
      self.mac = EthAddr(mac)    # set the mac address
      self.port = port

    def __str__(self):
      return','.join([str(self.ip), str(self.mac), str(self.port)])

  def __init__ (self, connection):    # 
    self.connection = connection
    self.listenTo(connection)
    # Initialize the server list
    self.servers = [
      self.Server('10.0.0.1', '00:00:00:00:00:01', 1),
      self.Server('10.0.0.2', '00:00:00:00:00:02', 2)]    # server 
    self.last_server = 0

  def get_next_server (self):    #
    # Round-robin load the servers
    self.last_server = (self.last_server + 1) % len(self.servers)
    return self.servers[self.last_server]

  def handle_arp (self, packet, in_port):    # function No.2

    # Get the ARP request from packet
    arp_req = packet.next

    # Create ARP reply
    arp_rep = arp()
    arp_rep.opcode = arp.REPLY
    arp_rep.hwsrc = LOAD_BALANCER_MAC    # hardware source
    arp_rep.hwdst = arp_req.hwsrc    # hardware destination
    arp_rep.protosrc = LOAD_BALANCER_IP    # software source
    arp_rep.protodst = arp_req.protosrc    # software destination

    # Create the Ethernet packet
    eth = ethernet()
    eth.type = ethernet.ARP_TYPE
    eth.dst = packet.src
    eth.src = LOAD_BALANCER_MAC
    eth.set_payload(arp_rep)

    # Send the ARP reply to client
    msg = of.ofp_packet_out()
    msg.data = eth.pack()
    msg.actions.append(of.ofp_action_output(port = of.OFPP_IN_PORT))
    msg.in_port = in_port
    self.connection.send(msg)

  def handle_request (self, packet, event):    # function No.3

    # Get the next server to handle the request
    server = self.get_next_server()

    "First install the reverse rule from server to client"

    msg = of.ofp_flow_mod()
    msg.idle_timeout = IDLE_TIMEOUT
    msg.hard_timeout = HARD_TIMEOUT
    msg.buffer_id = None

    # Set packet matching
    # Match (in_port, src MAC, dst MAC, src IP, dst IP)
    msg.match.in_port = server.port
    msg.match.dl_src = server.mac
    msg.match.dl_dst = packet.src
    msg.match.dl_type = ethernet.IP_TYPE
    msg.match.nw_src = server.ip
    msg.match.nw_dst = packet.next.srcip

    # Append actions
    # Set the src IP and MAC to load balancer's
    # Forward the packet to client's port
    msg.actions.append(of.ofp_action_nw_addr.set_src(LOAD_BALANCER_IP))
    msg.actions.append(of.ofp_action_dl_addr.set_src(LOAD_BALANCER_MAC))
    msg.actions.append(of.ofp_action_output(port = event.port))

    self.connection.send(msg)

    "Second install the forward rule from client to server"
    msg = of.ofp_flow_mod()
    msg.idle_timeout = IDLE_TIMEOUT
    msg.hard_timeout = HARD_TIMEOUT
    msg.buffer_id = None
    msg.data = event.ofp # Forward the incoming packet

    # Set packet matching
    # Match (in_port, MAC src, MAC dst, IP src, IP dst)
    msg.match.in_port = event.port
    msg.match.dl_src = packet.src
    msg.match.dl_dst = LOAD_BALANCER_MAC
    msg.match.dl_type = ethernet.IP_TYPE
    msg.match.nw_src = packet.next.srcip
    msg.match.nw_dst = LOAD_BALANCER_IP
    
    # Append actions
    # Set the dst IP and MAC to load balancer's
    # Forward the packet to server's port
    msg.actions.append(of.ofp_action_nw_addr.set_dst(server.ip))
    msg.actions.append(of.ofp_action_dl_addr.set_dst(server.mac))
    msg.actions.append(of.ofp_action_output(port = server.port))

    self.connection.send(msg)

    log.info("Installing %s <-> %s" % (packet.next.srcip, server.ip))

  def _handle_PacketIn (self, event):    # function No.4 

    if packet.type == packet.LLDP_TYPE or packet.type == packet.IPV6_TYPE:
      # Drop LLDP packets 
      # Drop IPv6 packets
      # send of command without actions

      msg = of.ofp_packet_out()
      msg.buffer_id = event.ofp.buffer_id
      msg.in_port = event.port
      self.connection.send(msg)

    elif packet.type == packet.ARP_TYPE:
      # Handle ARP request for load balancer

      # Only accept ARP request for load balancer
      if packet.next.protodst != LOAD_BALANCER_IP:
        return

      log.debug("Receive an ARP request")
      self.handle_arp(packet, event.port)

    elif packet.type == packet.IP_TYPE:
      # Handle client's request

      # Only accept ARP request for load balancer
      if packet.next.dstip != LOAD_BALANCER_IP:
        return

      log.debug("Receive an IPv4 packet from %s" % packet.next.srcip)
      self.handle_request(packet, event)


class load_balancer (EventMixin):    # my component

  def __init__ (self):
    self.listenTo(core.openflow)

  def _handle_ConnectionUp (self, event):
    log.debug("Connection %s" % event.connection)
    LoadBalancer(event.connection)


def launch ():    # registering component
  # Start load balancer
  core.registerNew(load_balancer)    # load_balancer 
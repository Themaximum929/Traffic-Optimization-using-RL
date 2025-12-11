import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
import json
import os

class TunnelNetworkVisualizer:
    def __init__(self, base_dir=r'D:\MAIE5532_Resources'):
        self.base_dir = base_dir
        
    def load_proposal_metrics(self):
        metrics_file = os.path.join(self.base_dir, 'Scripts', 'proposal_metrics_complete.json')
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def plot_significant_tunnel_network(self):
        """Visualize the significant tunnel nodes and their correlations"""
        metrics = self.load_proposal_metrics()
        if not metrics:
            print("No metrics data available")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Define significant node sequences
        sequences = {
            'WHT_SB_Kowloon': ['AID03204', 'AID03205', 'AID03206', 'AID03207', 'AID03208', 'AID03209', 'AID03210', 'AID03211'],
            'WHT_SB_HK': ['AID04221', 'AID04104'],
            'WHT_NB': ['AID04218', 'AID04219', 'AID03103'],
            'CHT_NB_R1': ['AID01108', 'AID01109', 'AID01110'],
            'CHT_NB_R2': ['TDSIEC10002', 'TDSIEC10003', 'TDSIEC10004'],
            'CHT_SB_R1': ['AID01208', 'AID01209', 'AID01211', 'AID01212', 'AID01213'],
            'CHT_SB_R2': ['AID05224', 'AID05225', 'AID05226'],
            'EHT_NB_R1': ['AID04210', 'AID04212'],
            'EHT_NB_R2': ['AID04106', 'AID04107', 'AID04122', 'AID04110'],
            'EHT_SB': ['AID02204', 'AID02205', 'AID02206', 'AID02207', 'AID02208', 'AID02209', 'AID02210', 'AID02211', 'AID02212', 'AID02213', 'AID02214']
        }
        
        # Color scheme for tunnels
        colors = {
            'WHT': '#FF6B6B',  # Red
            'CHT': '#4ECDC4',  # Teal
            'EHT': '#45B7D1'   # Blue
        }
        
        # Position nodes
        y_positions = {
            'WHT_SB_Kowloon': 8, 'WHT_SB_HK': 7, 'WHT_NB': 6,
            'CHT_NB_R1': 5, 'CHT_NB_R2': 4, 'CHT_SB_R1': 3, 'CHT_SB_R2': 2,
            'EHT_NB_R1': 1, 'EHT_NB_R2': 0, 'EHT_SB': -1
        }
        
        # Draw sequences
        for seq_name, nodes in sequences.items():
            tunnel = seq_name.split('_')[0]
            color = colors[tunnel]
            y = y_positions[seq_name]
            
            # Draw nodes
            for i, node in enumerate(nodes):
                x = i * 2
                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.7)
                ax.add_patch(circle)
                ax.text(x, y, node.replace('AID', '').replace('TDSIEC', 'T'), 
                       ha='center', va='center', fontsize=8, fontweight='bold')
                
                # Draw arrows between nodes
                if i < len(nodes) - 1:
                    ax.arrow(x + 0.3, y, 1.4, 0, head_width=0.1, head_length=0.1, 
                            fc=color, ec=color, alpha=0.8)
            
            # Add sequence label
            ax.text(-1, y, seq_name.replace('_', ' '), ha='right', va='center', 
                   fontweight='bold', color=color)
        
        # Draw inter-tunnel connections
        inter_connections = [
            ('AID04104', 'TDSIEC10004'),
            ('TDSIEC10004', 'AID04106'),
            ('AID03210', 'AID05224')
        ]
        
        # Find positions for inter-connections
        node_positions = {}
        for seq_name, nodes in sequences.items():
            y = y_positions[seq_name]
            for i, node in enumerate(nodes):
                node_positions[node] = (i * 2, y)
        
        for start, end in inter_connections:
            if start in node_positions and end in node_positions:
                x1, y1 = node_positions[start]
                x2, y2 = node_positions[end]
                ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.5, linewidth=2)
                ax.text((x1+x2)/2, (y1+y2)/2, '→', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        
        # Add toll price analysis if available
        toll_analysis = metrics.get('toll_plan_analysis', {}).get('toll_price_analysis', {})
        if toll_analysis:
            price_corr = toll_analysis.get('toll_price_correlation', {})
            if price_corr:
                high_toll = price_corr.get('high_toll_periods', {})
                low_toll = price_corr.get('low_toll_periods', {})
                
                # Add toll effectiveness box
                toll_text = f"Toll Effectiveness:\n$60 periods: {high_toll.get('avg_speed', 0):.1f} km/h\n$20 periods: {low_toll.get('avg_speed', 0):.1f} km/h"
                ax.text(0.02, 0.98, toll_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                       verticalalignment='top', fontsize=10)
        
        ax.set_xlim(-3, 20)
        ax.set_ylim(-2, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Significant Tunnel Network Topology\n(Peak Hours 7-10 AM, 17-20 PM, Occupancy >50%)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [plt.Circle((0, 0), 0.3, color=colors['WHT'], alpha=0.7, label='WHT'),
                          plt.Circle((0, 0), 0.3, color=colors['CHT'], alpha=0.7, label='CHT'),
                          plt.Circle((0, 0), 0.3, color=colors['EHT'], alpha=0.7, label='EHT')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        output_path = os.path.join(self.base_dir, 'Scripts', 'significant_tunnel_network.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Significant tunnel network saved: {output_path}")

if __name__ == "__main__":
    visualizer = TunnelNetworkVisualizer()
    visualizer.plot_significant_tunnel_network()
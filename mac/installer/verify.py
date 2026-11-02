"""Check installer config; with a .pkg argument, expand and verify its real payload (no install)."""
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
APP = Path('Applications/CleanMyMic.app')
DRIVER = Path('Library/Audio/Plug-Ins/HAL/CleanMyMic.driver')
SUPPORT = Path('Library/Application Support/CleanMyMic')
BASE_HASH = 'a8896e66f673dd54b7251577806d51685d2def46c918310c73a95ca6319373bc'
DRIVER_REV = '574029f0d3175ccf29a826d74be4da3967b1cb92'


def check_distribution(path):
    dist = ET.parse(path).getroot()
    assert dist.find('options').get('hostArchitectures') == 'arm64'
    domains = dist.find('domains').attrib
    assert domains == dict(enable_anywhere='false', enable_currentUserHome='false', enable_localSystem='true')
    assert dist.find('volume-check/allowed-os-versions/os-version').get('min') == '15.0'
    assert dist.find("pkg-ref[@onConclusion='RequireRestart']") is not None
    assert dist.find('pkg-ref/must-close/app').get('id') == 'dev.vaze.CleanMyMic'


def check_config():
    check_distribution(HERE / 'Distribution.xml')
    with (HERE / 'components.plist').open('rb') as file:
        bundles = plistlib.load(file)
    assert {b['RootRelativeBundlePath'] for b in bundles} == {str(APP), str(DRIVER)}
    for bundle in bundles:
        assert not bundle['BundleIsRelocatable']
        assert bundle['BundleHasStrictIdentifier'] and bundle['BundleIsVersionChecked']
        assert bundle['BundleOverwriteAction'] == 'upgrade'
    with (HERE / 'entitlements.plist').open('rb') as file:
        assert plistlib.load(file) == {'com.apple.security.device.audio-input': True}
    subprocess.run(['bash', '-n', str(HERE / 'build.sh')], check=True)
    subprocess.run(['sh', '-n', str(HERE / 'uninstall.sh')], check=True)


def check_package(package):
    with tempfile.TemporaryDirectory(prefix='cleanmymic-pkg-') as temp:
        expanded = Path(temp) / 'expanded'
        subprocess.run(['pkgutil', '--expand-full', str(package), str(expanded)], check=True)
        check_distribution(expanded / 'Distribution')
        components = list(expanded.glob('*.pkg'))
        assert len(components) == 1
        component = components[0]
        info = ET.parse(component / 'PackageInfo').getroot()
        assert info.get('identifier') == 'dev.vaze.CleanMyMic.pkg'
        assert info.get('install-location') == '/'
        assert not (component / 'Scripts').exists(), 'No privileged install scripts are needed'
        root = component / 'Payload'
        for file in root.rglob('*'):
            if file.is_file() or file.is_symlink():
                assert any(file.is_relative_to(root / allowed) for allowed in (APP, DRIVER, SUPPORT)), file
                assert not file.is_symlink(), file
        for relative, identifier in ((APP, 'dev.vaze.CleanMyMic'), (DRIVER, 'dev.vaze.CleanMyMic.driver')):
            bundle = root / relative
            with (bundle / 'Contents/Info.plist').open('rb') as file:
                metadata = plistlib.load(file)
            assert metadata['CFBundleIdentifier'] == identifier
            executable = bundle / 'Contents/MacOS' / metadata['CFBundleExecutable']
            subprocess.run(['lipo', str(executable), '-verify_arch', 'arm64'], check=True)
            subprocess.run(['codesign', '--verify', '--deep', '--strict', str(bundle)], check=True)
            if relative == APP:
                assert metadata['LSMinimumSystemVersion'] == '15.0'
                assert metadata['NSMicrophoneUsageDescription']
                assert info.get('version') == f"{metadata['CFBundleShortVersionString']}.{metadata['CFBundleVersion']}"
                entitlements = subprocess.check_output(['codesign', '-d', '--entitlements', ':-', str(bundle)], stderr=subprocess.DEVNULL)
                assert plistlib.loads(entitlements) == {'com.apple.security.device.audio-input': True}
                signature = subprocess.run(['codesign', '-d', '--verbose=4', str(bundle)], capture_output=True, text=True, check=True)
                assert 'runtime' in signature.stderr
        models = list((root / APP / 'Contents/Resources').glob('*.mlmodelc'))
        assert [model.name for model in models] == ['Hold-10k.mlmodelc']
        manifest = json.loads((root / SUPPORT / 'build.json').read_text())
        assert manifest['model'] == 'CleanMyMic Base'
        assert manifest['driver_revision'] == DRIVER_REV
        assert hashlib.sha256((models[0] / 'weights/weight.bin').read_bytes()).hexdigest() == manifest['model_weights_sha256'] == BASE_HASH
        with tarfile.open(root / SUPPORT / 'driver-source.tar.gz') as source:
            names = source.getnames()
            assert not any('/.git/' in name or '..' in Path(name).parts for name in names)
            for suffix in ('mac/driver/build.sh', 'mac/AppIcon.icns', 'mac/driver/build/BlackHole/BlackHole/BlackHole.c', 'LICENSE.txt'):
                assert f'CleanMyMic-driver-source/{suffix}' in names
            driver = source.extractfile('CleanMyMic-driver-source/mac/driver/build/BlackHole/BlackHole/BlackHole.c').read()
            assert b'= kAudioDeviceTransportTypeUSB;' in driver
        for name in ('LICENSE', 'licenses/BlackHole.txt', 'licenses/DeepFilterNet-MIT.txt', 'uninstall.sh'):
            assert (root / SUPPORT / name).stat().st_size > 0
        assert (root / SUPPORT / 'uninstall.sh').stat().st_mode & 0o111
        print('Package verified: app, driver, Base weights, source, notices, architecture, signatures, destinations, restart.')


if __name__ == '__main__':
    check_config()
    if len(sys.argv) == 2:
        check_package(Path(sys.argv[1]).resolve())
    else:
        assert len(sys.argv) == 1, 'Usage: verify.py [installer.pkg]'
        print('Installer configuration verified.')
